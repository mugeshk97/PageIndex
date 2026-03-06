import os
import json
import re
import asyncio
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from page_index_md import md_to_tree

load_dotenv()
# -------------------------------
# Azure Clients
# -------------------------------

def get_docintel_client():

    endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
    key = os.getenv("AZURE_DOC_INTEL_KEY")

    if key:
        credential = AzureKeyCredential(key)
    else:
        credential = DefaultAzureCredential()

    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=credential
    )


def get_openai_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")

    if key:
        return AsyncAzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )

    credential = DefaultAzureCredential()
    scope = "https://cognitiveservices.azure.com/.default"

    def token_provider():
        token = credential.get_token(scope)
        return token.token

    return AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=endpoint,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


# -------------------------------
# PDF → Markdown
# -------------------------------

def pdf_to_markdown(pdf_path, md_path):

    client = get_docintel_client()

    with open(pdf_path, "rb") as f:

        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output_content_format=DocumentContentFormat.MARKDOWN
        )

    result = poller.result()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.content)

    print("Markdown saved:", md_path)


# -------------------------------
# LLM Call
# -------------------------------

async def call_llm(prompt):

    client = get_openai_client()

    response = await client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# -------------------------------
# Remove Heavy Fields
# -------------------------------

def remove_fields(data, fields=["text"]):

    if isinstance(data, dict):
        return {
            k: remove_fields(v, fields)
            for k, v in data.items()
            if k not in fields
        }

    if isinstance(data, list):
        return [remove_fields(i, fields) for i in data]

    return data


# -------------------------------
# Create Node Mapping
# -------------------------------

def get_all_nodes(tree):

    if isinstance(tree, dict):

        nodes = [tree]

        for child in tree.get("nodes", []):
            nodes.extend(get_all_nodes(child))

        return nodes

    elif isinstance(tree, list):

        nodes = []

        for item in tree:
            nodes.extend(get_all_nodes(item))

        return nodes

    return []


def create_node_mapping(tree):

    nodes = get_all_nodes(tree)

    return {
        node["node_id"]: node
        for node in nodes
        if node.get("node_id")
    }


# -------------------------------
# Retrieval — Strategy 1: Keyword / Title Matching
# -------------------------------

def keyword_match_nodes(node_mapping, query):
    """
    Title-based matching with no LLM cost.
    Scores each node by the fraction of query terms (>3 chars) found in its title.
    Returns all node_ids that matched at least one term, sorted by score descending.
    """
    query_terms = [t.lower() for t in re.split(r'\W+', query) if len(t) > 3]
    if not query_terms:
        return []

    matched = []
    for node_id, node in node_mapping.items():
        title_lower = node['title'].lower()
        match_count = sum(1 for t in query_terms if t in title_lower)
        if match_count > 0:
            matched.append((node_id, match_count / len(query_terms)))

    matched.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in matched]


# -------------------------------
# Retrieval — Strategy 2: LLM with Summaries
# -------------------------------

def _strip_json_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text.rstrip())
    return text.strip()


async def llm_retrieve_with_summaries(structure, query):
    """
    LLM-based retrieval over the tree skeleton (summaries kept, text + line_num stripped).
    Asks the model to search EXHAUSTIVELY across every branch and return all matches
    with confidence scores. Only nodes with confidence >= 0.75 are returned.
    """
    skeleton = remove_fields(structure, ["text", "line_num"])
    skeleton_json = json.dumps(skeleton, indent=2)

    if len(skeleton_json) > 60000:
        print(f"[WARN] Skeleton is {len(skeleton_json)} chars — may approach context limits")

    prompt = f"""You are searching a document tree to find ALL sections relevant to a query.
The content may be scattered across the document — search EVERY branch exhaustively.

Query: "{query}"

Rules:
- A node is relevant if it DIRECTLY contains the queried information (not just tangentially).
- Check both the node title AND its summary / prefix_summary.
- Include ALL occurrences even if their titles differ slightly or they appear in different parts.
- Assign confidence 0.0–1.0. Only include nodes with confidence >= 0.75.
- Prefer specific child nodes over their parent when only the child is relevant.

Document tree (summaries shown, full text omitted):
{skeleton_json}

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{
  "thinking": "<check every branch; explain which nodes contain the queried content and why>",
  "nodes": [
    {{"node_id": "<id>", "confidence": <0.0-1.0>, "reason": "<why this node is relevant>"}}
  ]
}}"""

    response = await call_llm(prompt)
    try:
        result = json.loads(_strip_json_fences(response))
        return [
            n["node_id"]
            for n in result.get("nodes", [])
            if float(n.get("confidence", 0)) >= 0.75
        ]
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"[WARN] llm_retrieve_with_summaries parse failed. Raw: {response[:200]}")
        return []


# -------------------------------
# Retrieval — Strategy 3: Content Verification
# -------------------------------

async def _verify_one(node_id, node, query):
    """Verify a single candidate node against its actual text content."""
    content = node.get("text") or node.get("summary") or node.get("prefix_summary") or ""
    if not content.strip():
        return None

    prompt = f"""Does the following document section contain information about "{query}"?

Section title: {node['title']}
Section content:
{content[:3000]}

Respond ONLY with valid JSON (no markdown fences):
{{"relevant": true or false, "confidence": <0.0-1.0>}}"""

    response = await call_llm(prompt)
    try:
        result = json.loads(_strip_json_fences(response))
        if result.get("relevant") and float(result.get("confidence", 0)) >= 0.7:
            return node_id
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"[WARN] _verify_one parse failed for node {node_id}. Raw: {response[:200]}")
    return None


async def verify_nodes(candidates, node_mapping, query, batch_size=5):
    """Verify candidate nodes in batches to avoid API rate limits."""
    verified = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        tasks = [
            _verify_one(nid, node_mapping[nid], query)
            for nid in batch
            if nid in node_mapping
        ]
        results = await asyncio.gather(*tasks)
        verified.extend(r for r in results if r is not None)
    return verified


# -------------------------------
# Combined Retrieval
# -------------------------------

async def retrieve_nodes(structure, query, node_mapping):
    """
    Three-strategy retrieval:
      1. Keyword matching on titles  — fast, zero LLM cost, reliable for exact/near-exact titles
      2. LLM retrieval with summaries — semantic, catches scattered/differently-titled sections
      3. Content verification          — removes false positives by checking actual text

    Union of 1+2 gives high recall; verification gates precision.
    """
    keyword_matches = set(keyword_match_nodes(node_mapping, query))
    print(f"Keyword matches: {sorted(keyword_matches)}")

    llm_matches = set(await llm_retrieve_with_summaries(structure, query))
    print(f"LLM matches: {sorted(llm_matches)}")

    candidates = list(keyword_matches | llm_matches)
    print(f"Combined candidates ({len(candidates)}): {sorted(candidates)}")

    verified = await verify_nodes(candidates, node_mapping, query)
    print(f"Verified nodes ({len(verified)}): {sorted(verified)}")

    return verified


# -------------------------------
# Extract Sections
# -------------------------------

def extract_sections(node_ids, node_mapping):

    sections = []

    for nid in node_ids:

        node = node_mapping.get(nid)

        if not node:
            continue

        sections.append({
            "node_id": nid,
            "title": node["title"],
            "text": node.get("text")
        })

    return sections

# -------------------------------
# Main Pipeline
# -------------------------------

async def run_pipeline(pdf_path, query):

    md_path = "document.md"
    tree_path = "tree.json"

    # Step 1: PDF → Markdown
    pdf_to_markdown(pdf_path, md_path)

    # Step 2: Markdown → Tree
    # if_add_node_text="yes" so text is available for verification (step 4) and extraction (step 5)
    tree = await md_to_tree(
        md_path=md_path,
        if_thinning=False,
        if_add_node_summary="yes",
        if_add_node_text="yes",
        summary_token_threshold=200,
        model=AZURE_DEPLOYMENT
    )

    json.dump(tree, open(tree_path, "w"), indent=2)
    print("Tree created")

    structure = tree["structure"]

    # Step 3: Build node mapping before retrieval (needed by all three strategies)
    node_mapping = create_node_mapping(structure)

    # Step 4: Retrieve Nodes — keyword + LLM + verification
    node_ids = await retrieve_nodes(structure, query, node_mapping)
    print("Verified relevant nodes:", node_ids)

    # Step 5: Extract Sections
    sections = extract_sections(node_ids, node_mapping)

    print("\nRetrieved Sections:\n")

    for s in sections:
        print("----", s["title"])
        print(s["text"])


# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":

    pdf_path = r"c:\Users\KANNAMU5\Downloads\Final Assets\Final Assets\fa-11419769-fab-fabhalta-igan-patient-understanding-your-igan-digital-pi-update-3-25 - edited ISI1.pdf"

    query = "Important Safety Informations"

    asyncio.run(run_pipeline(pdf_path, query))