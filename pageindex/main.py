import os
import json
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
# Retrieve Relevant Nodes
# -------------------------------

async def retrieve_nodes(tree, query):

    tree_for_llm = remove_fields(tree, ["text"])

    prompt = f"""
    You are given a query and the tree structure of a document.
    You need to find all nodes that are likely to contain the answer.

    Query: {query}

    Document tree structure: {json.dumps(tree_for_llm)}

    Reply in the following JSON format:
    {{
    "thinking": <your reasoning about which nodes are relevant>,
    "node_list": [node_id1, node_id2, ...]
    }}
    """

    response = await call_llm(prompt)

    return json.loads(response)["node_list"]


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
    tree = await md_to_tree(
        md_path=md_path,
        if_thinning=False,
        if_add_node_summary="yes",
        summary_token_threshold=200,
        model="gpt-4o"
    )

    json.dump(tree, open(tree_path, "w"), indent=2)

    print("Tree created")

    # Step 3: Retrieve Nodes
    node_ids = await retrieve_nodes(tree, query)

    print("Relevant Nodes:", node_ids)

    # Step 4: Extract Sections
    node_mapping = create_node_mapping(tree)

    sections = extract_sections(node_ids, node_mapping)

    print("\nRetrieved Sections:\n")

    for s in sections:
        print("----", s["title"])
        print(s["text"])

    # # Step 5: Answer
    # answer = await answer_question(sections, query)

    # print("\nAnswer:\n")
    # print(answer)


# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":

    pdf_path = r"c:\Users\KANNAMU5\Downloads\Final Assets\Final Assets\fa-11419769-fab-fabhalta-igan-patient-understanding-your-igan-digital-pi-update-3-25 - edited ISI1.pdf"

    query = "Important Safety Informations"

    asyncio.run(run_pipeline(pdf_path, query))