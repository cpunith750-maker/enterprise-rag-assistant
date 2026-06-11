from pathlib import Path


def load_documents(data_dir="data/sample", max_files_per_source=None):
    """
    Loads text documents from data/sample.

    Expected folder structure:
    data/sample/
        confluence/
        jira/
        github/
        google_drive/
    """

    documents = []
    base_path = Path(data_dir)

    source_folders = ["confluence", "jira", "github", "google_drive"]

    for source_type in source_folders:
        source_path = base_path / source_type

        if not source_path.exists():
            print(f"Warning: folder not found: {source_path}")
            continue

        files = list(source_path.glob("*.txt"))

        if max_files_per_source:
            files = files[:max_files_per_source]

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

                if not text:
                    continue

                document = {
                    "source_type": source_type,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "text": text,
                }

                documents.append(document)

            except Exception as e:
                print(f"Could not read file: {file_path}. Error: {e}")

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Total documents loaded: {len(docs)}")

    if docs:
        print("\nSample document:")
        print("Source:", docs[0]["source_type"])
        print("File name:", docs[0]["file_name"])
        print("Text preview:")
        print(docs[0]["text"][:500])
    else:
        print("No documents loaded.")