import glob
import os

from huggingface_hub import upload_file

card_files = glob.glob("cards/*.md")

for card_file in card_files:
    file_basename = os.path.basename(card_file).split(".")[0]

    # create a temporary file with the header and contents of the markdown file
    with open(card_file, "r") as f:
        contents = f.read()
    temp_file_contents = f"""
# {file_basename}

See https://confirmlabs.org/posts/catalog.html for details.

{contents}
"""

    with open("tempfile.md", "w") as f:
        f.write(temp_file_contents)

    # upload the temporary file to huggingface_hub
    upload_file(
        path_or_fileobj="tempfile.md",
        path_in_repo="README.md",
        repo_id=f"Confirm-Labs/pile_{file_basename}",
        repo_type="dataset",
    )

os.remove("tempfile.md")
