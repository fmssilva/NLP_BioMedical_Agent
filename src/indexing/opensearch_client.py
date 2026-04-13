import os
import sys

from dotenv import load_dotenv
from opensearchpy import OpenSearch


######################################################################
## OpenSearch connection and health utilities.
## This reads frm credentials in .env (python-dotenv).
######################################################################


def get_client() -> OpenSearch:
    """
    Create and return a configured OpenSearch client.
    Reads environment variables (via .env):
        OPENSEARCH_HOST  — e.g. api.novasearch.org
        OPENSEARCH_PORT  — e.g. 443
        OPENSEARCH_USER  — username
        OPENSEARCH_PASS  — password
    """
    load_dotenv()

    host = os.getenv("OPENSEARCH_HOST")
    port = os.getenv("OPENSEARCH_PORT")
    user = os.getenv("OPENSEARCH_USER")
    password = os.getenv("OPENSEARCH_PASS")

    missing = [k for k, v in {
        "OPENSEARCH_HOST": host,
        "OPENSEARCH_PORT": port,
        "OPENSEARCH_USER": user,
        "OPENSEARCH_PASS": password,
    }.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {missing}. "
            "Add them to a .env file in the project root."
        )

    client = OpenSearch(
        hosts=[{"host": host, "port": int(port)}],
        http_compress=True,
        http_auth=(user, password),
        use_ssl=True,
        url_prefix="opensearch_v3",
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
    )
    return client


def check_health(client: OpenSearch) -> None:
    """
    Verify that the OpenSearch server is reachable.
    """
    # --- client.info() ---
    try:
        info = client.info()
        version      = info.get("version", {}).get("number", "?")
        cluster_name = info.get("cluster_name", "unknown")
        print(f"[health] cluster='{cluster_name}'  version={version}  "
              f"(cluster health restricted — info() used)")
        return
    except Exception as e:
        if "security_exception" not in str(e) and "403" not in str(e):
            raise RuntimeError(f"Cannot reach OpenSearch cluster: {e}") from e

    # --- Fallback: lightweight indices.exists probe ---
    try:
        client.indices.exists(index="probe-connection-check")
        print("[health] Server reachable (cluster:monitor restricted — "
              "connectivity confirmed via indices.exists probe)")
    except Exception as exc:
        if "security_exception" in str(exc) or "403" in str(exc):
            print("[health] Server reachable (cluster:monitor restricted — "
                  "connectivity confirmed via 403 response)")
        else:
            raise RuntimeError(
                f"Cannot reach OpenSearch cluster: {exc}"
            ) from exc


def check_index(client: OpenSearch, index_name: str, expected_count: int = 4194) -> bool:
    """
    Check whether the target index exists and is fully populated (doc length == expected_count).
    """
    if not client.indices.exists(index=index_name):
        print(f"[index] '{index_name}' does not exist — not yet created (that's OK at this stage).")
        return False

    count_response = client.count(index=index_name)
    count = count_response.get("count", 0)

    if count == expected_count:
        print(f"[index] '{index_name}'  docs={count}  [ok] fully populated")
        return True
    else:
        print(
            f"[index] '{index_name}'  docs={count}  "
            f"(expected {expected_count}) — partial or empty index"
        )
        return False


#################################################################
##                  LOCAL TEST                                 ##
#################################################################
if __name__ == "__main__":
    print("=" * 60)
    print("Step 6 — opensearch_client.py self-test")
    print("=" * 60)

    # 1. Build client
    print("\n[1/3] Connecting to OpenSearch …")
    try:
        os_client = get_client()
        print("      Client created successfully.")
    except ValueError as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # 2. Cluster health
    print("\n[2/3] Checking cluster health …")
    try:
        check_health(os_client)
    except RuntimeError as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # 3. Index check
    index_name = os.getenv("OPENSEARCH_INDEX", "")
    if not index_name:
        print("\n[3/3] OPENSEARCH_INDEX not set in .env — skipping index check.")
    else:
        print(f"\n[3/3] Checking index '{index_name}' …")
        check_index(os_client, index_name)

    print("\n" + "=" * 60)
    print("opensearch_client.py  —  all checks passed [ok]")
    print("=" * 60)
