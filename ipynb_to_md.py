"""
Converts ipynb file to markdown file and stores outputs in a separate folder.
"""

import os
import subprocess
import shutil
import argparse


def convert_notebooks_to_markdown(root_dir, export_dir):
    """Converts ipynb file to markdown

    Args:
        root_dir (str): root dir to start converting from
        export_dir (str): where to save converted markdown files
    """
    os.makedirs(export_dir, exist_ok=True)  # Create export folder if missing

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.startswith("."):
                notebook_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(dirpath, root_dir)

                # Prepare output subdirectory
                output_subdir = os.path.join(export_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)

                base_name = os.path.splitext(filename)[0]
                md_file = os.path.join(output_subdir, f"{base_name}.md")
                files_folder = os.path.join(output_subdir, f"{base_name}_files")

                # Remove old outputs if exist
                if os.path.exists(md_file):
                    os.remove(md_file)
                    print(f"🗑️ Removed old: {md_file}")
                if os.path.isdir(files_folder):
                    shutil.rmtree(files_folder)
                    print(f"🗑️ Removed old folder: {files_folder}")

                # Convert to markdown in-place, then move outputs
                print(f"🔄 Converting: {notebook_path}")
                try:
                    subprocess.run(
                        [
                            "jupyter",
                            "nbconvert",
                            "--to",
                            "markdown",
                            "--output",
                            base_name,
                            "--output-dir",
                            output_subdir,
                            notebook_path,
                        ],
                        check=True,
                    )
                    print(f"✅ Done: {md_file}\n")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error converting {notebook_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all Jupyter Notebooks to Markdown and "
        "store in a separate export folder."
    )
    parser.add_argument(
        "folder", type=str, help="Path to the root folder containing .ipynb files"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="markdown-version",
        help="Path to the folder where markdown files should be saved",
    )
    args = parser.parse_args()
    convert_notebooks_to_markdown(args.folder, args.export)
