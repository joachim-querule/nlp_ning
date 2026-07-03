from pathlib import Path
import gradio_client

utils_path = Path(gradio_client.__file__).parent / "utils.py"

text = utils_path.read_text(encoding="utf-8")
lines = text.splitlines()

already_patched = "if isinstance(schema, bool):" in text

if already_patched:
    print("Patch déjà présent.")
    print("Fichier :", utils_path)
else:
    for i, line in enumerate(lines):
        if line.startswith("def get_type("):
            indent = "    "
            lines.insert(i + 1, indent + 'if isinstance(schema, bool):')
            lines.insert(i + 2, indent + '    return "boolean"')
            break
    else:
        raise RuntimeError("Impossible de trouver la fonction get_type dans utils.py")

    utils_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Patch appliqué avec succès.")
    print("Fichier modifié :", utils_path)
