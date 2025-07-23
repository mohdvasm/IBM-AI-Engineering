"""
Converts ipynb file to markdown file
"""

import os
import subprocess
import shutil
import argparse


def convert_notebooks_to_markdown(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.startswith("."):
                notebook_path = os.path.join(dirpath, filename)
                base_name = os.path.splitext(filename)[0]
                md_file = os.path.join(dirpath, f"{base_name}.md")
                files_folder = os.path.join(dirpath, f"{base_name}_files")

                # Delete old .md file if exists
                if os.path.exists(md_file):
                    os.remove(md_file)
                    print(f"🗑️ Removed old: {md_file}")

                # Delete old *_files/ folder if exists
                if os.path.isdir(files_folder):
                    shutil.rmtree(files_folder)
                    print(f"🗑️ Removed old folder: {files_folder}")

                # Convert to markdown
                print(f"🔄 Converting: {notebook_path}")
                try:
                    subprocess.run(
                        ["jupyter", "nbconvert", "--to", "markdown", notebook_path],
                        check=True,
                    )
                    print(f"✅ Done: {md_file}\n")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error converting {notebook_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all Jupyter Notebooks in a folder to Markdown with image output."
    )
    parser.add_argument(
        "folder", type=str, help="Path to the root folder containing .ipynb files"
    )
    args = parser.parse_args()
    convert_notebooks_to_markdown(args.folder)
