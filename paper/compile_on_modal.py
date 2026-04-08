"""Compile NexusQuant paper on Modal (no local LaTeX needed)."""
import modal
import os

app = modal.App("nexusquant-paper-compile")

paper_dir = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("texlive-latex-base", "texlive-latex-extra", "texlive-fonts-recommended",
                 "texlive-fonts-extra", "texlive-science", "latexmk", "cm-super")
    .add_local_dir(paper_dir, remote_path="/root/paper")
)


@app.function(image=image, timeout=300)
def compile_paper():
    import subprocess
    os.chdir("/root/paper")

    def run_pdflatex():
        return subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "nexusquant.tex"],
            capture_output=True, text=True, timeout=120
        )

    # Correct order: pdflatex -> bibtex -> pdflatex -> pdflatex
    print("Pass 1: pdflatex...")
    run_pdflatex()

    print("Pass 2: bibtex...")
    bib_r = subprocess.run(["bibtex", "nexusquant"], capture_output=True, text=True, timeout=30)
    for line in bib_r.stdout.split('\n'):
        if line.strip():
            print(f"  bibtex: {line}")

    print("Pass 3: pdflatex...")
    run_pdflatex()

    print("Pass 4: pdflatex (final)...")
    r = run_pdflatex()
    # Print warnings from final pass
    for line in r.stdout.split('\n'):
        if 'undefined' in line.lower() or line.startswith('!'):
            print(f"  WARN: {line}")

    if os.path.exists("nexusquant.pdf"):
        with open("nexusquant.pdf", "rb") as f:
            pdf_bytes = f.read()
        print(f"PDF generated: {len(pdf_bytes)} bytes ({len(pdf_bytes)/1024:.0f} KB)")
        return pdf_bytes
    else:
        print("PDF generation FAILED")
        # Print full log
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "nexusquant.tex"],
            capture_output=True, text=True, timeout=120
        )
        for line in r.stdout.split('\n')[-50:]:
            print(line)
        return None


@app.local_entrypoint()
def main():
    print("Compiling paper on Modal...")
    pdf_bytes = compile_paper.remote()
    if pdf_bytes:
        out_path = os.path.join(paper_dir, "nexusquant.pdf")
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"PDF saved to {out_path} ({len(pdf_bytes)/1024:.0f} KB)")
    else:
        print("Compilation failed.")
