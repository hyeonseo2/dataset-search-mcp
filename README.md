# dataset-search-mcp

Unified **Model Context Protocol (MCP)** server for open-dataset discovery.
Search across **Hugging Face**, **Zenodo**, and optionally **Kaggle**, then generate ready-to-run **Colab starter code** for any result.

---

## Live demo

You can try the search & ranking logic in a simple UI here:
**[Open Dataset Finder (Hugging Face Spaces)](https://huggingface.co/spaces/Hyeonseo/Open-Dataset-Finder)**

---

## Features

* Multi-source search: Hugging Face / Zenodo / Kaggle (when credentials are available)
* Sensible ranking (BM25 + fuzzy + light recency weighting)
* Kaggle API with automatic CLI fallback
* Safe by default: the server returns metadata only (no server-side downloads)
* One-click starter snippets for quick experimentation

---

## Repository layout

```
dataset-search-mcp/
├─ src/
│  └─ dataset_search_mcp/
│     ├─ __init__.py
│     └─ server.py          # MCP server + tools
├─ examples/
│  ├─ claude-desktop.settings.json
│  └─ cursor.settings.json
├─ .github/workflows/
│  ├─ ci.yml
│  └─ release.yml
├─ Dockerfile
├─ pyproject.toml
├─ .dockerignore
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## Install (local)

> Requires Python 3.9+

```bash
pip install -e .
dataset-search-mcp
```

This starts the MCP server over **stdio** (awaiting an MCP client).

---

## Docker

### Build

```bash
docker build -t dataset-search-mcp:local .
```

### Quick smoke test (import only)

```bash
docker run --rm --entrypoint python dataset-search-mcp:local -c \
"import importlib; m=importlib.import_module('dataset_search_mcp.server'); print('OK', hasattr(m,'main'))"
# Expected: OK True
```

### Manual run (server waits for a client)

```bash
docker run -it --rm dataset-search-mcp:local
```

---

## Using with Claude Desktop

Add the server to **Settings → MCP Servers**.

**Simplest (Docker):**

```json
{
  "mcpServers": {
    "dataset-search-mcp": {
      "command": "docker",
      "args": ["run","-i","--rm","dataset-search-mcp:local"]
    }
  }
}
```

**With Kaggle credentials:**

```json
{
  "mcpServers": {
    "dataset-search-mcp": {
      "command": "docker",
      "args": [
        "run","-i","--rm",
        "-e","KAGGLE_USERNAME=your_username",
        "-e","KAGGLE_KEY=your_api_key",
        "dataset-search-mcp:local"
      ]
    }
  }
}
```

Restart Claude Desktop and open a **new chat**.

---

## Tools (overview)

### `search_datasets`

Search public datasets across the supported sources.

**Args (common):**

* `query` (string, required)
* `sources` (optional): e.g. `["huggingface","zenodo"]`
  *Note:* the server is tolerant—string forms like `"huggingface, zenodo"` also work.
* `limit` (optional, default 40): per-source cap before ranking
* `format_filter` (optional): e.g. `"csv"` or `"json"`

**Example call (as JSON):**

```json
{"query":"korean weather","sources":["huggingface","zenodo"],"limit":10}
```

**Returns:** a ranked array of items, each with:
`source, id, title, description, updated, url, download_url, formats, score`.

---

### `starter_code`

Generate a small Python snippet to quickly try the selected dataset in Colab.

**Args (typical):**

* `source` (e.g., `"huggingface"`, `"zenodo"`, `"kaggle"`)
* `id`
* `url` (optional)
* `download_url` (optional; if present and CSV, the snippet loads it directly)
* `formats` (optional; used to choose the best snippet)

---

## Kaggle credentials (brief)

Provide **either** environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

or a file:

```
~/.kaggle/kaggle.json
{"username":"your_username","key":"your_api_key"}
```

> Some Kaggle datasets require accepting terms on the website first.

---

## How it works (short)

* **Hugging Face:** `list_datasets()` plus optional `dataset_info()` for card details
* **Zenodo:** REST search via `GET /api/records`
* **Kaggle:** API first; fallback to CLI `datasets list --csv`
* **Ranking:** BM25 + fuzzy matching + light recency factor; duplicates merged on `(source,id)`

---

## Quick examples (in chat)

* Search HF + Zenodo:

```json
{"query":"korean weather","sources":["huggingface","zenodo"],"limit":10}
```

* CSV-only on Zenodo:

```json
{"query":"traffic accident Korea","sources":["zenodo"],"limit":20,"format_filter":"csv"}
```

* Then request starter code using one of the returned items’ fields (`source`, `id`, `url`, `download_url`, `formats`).
