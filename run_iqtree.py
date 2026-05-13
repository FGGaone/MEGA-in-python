#!/usr/bin/env python3
"""
IQ-TREE Runner Script
=====================
A Python wrapper to run IQ-TREE2 phylogenetic analysis with
configurable options for HIV/HBV genomic data.

Usage:
    python run_iqtree.py -i alignment.fasta [options]

Author: BHP Bioinformatics
"""

import argparse
import subprocess
import os
import sys
import logging
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Python wrapper for IQ-TREE2 phylogenetic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with automatic model selection:
  python run_iqtree.py -i alignment.fasta

  # Specify substitution model:
  python run_iqtree.py -i alignment.fasta -m GTR+G

  # With bootstrap replicates:
  python run_iqtree.py -i alignment.fasta -b 1000

  # With ultrafast bootstrap (recommended):
  python run_iqtree.py -i alignment.fasta -bb 1000

  # Full HIV analysis example:
  python run_iqtree.py -i hiv_alignment.fasta -m GTR+G+I -bb 1000 -nt AUTO -o outgroup_seq

  # Full HBV analysis example:
  python run_iqtree.py -i hbv_alignment.fasta -m HKY+G -bb 1000 -nt 4
        """
    )

    # Required
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input alignment file (FASTA, PHYLIP, or NEXUS format)"
    )

    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "-m", "--model",
        default="TEST",
        help="Substitution model. Use 'TEST' for automatic ModelTest (default: TEST)\n"
             "Common models: GTR+G, GTR+G+I, HKY+G, TrN+G, K80"
    )
    model_group.add_argument(
        "--mset",
        default=None,
        help="Restrict model testing to a set e.g. --mset GTR,HKY,TrN"
    )

    # Bootstrap
    boot_group = parser.add_argument_group("Bootstrap Options")
    boot_group.add_argument(
        "-bb", "--ultrafast-bootstrap",
        type=int,
        default=None,
        metavar="N",
        help="Ultrafast bootstrap replicates (recommended, min 1000). e.g. -bb 1000"
    )
    boot_group.add_argument(
        "-b", "--standard-bootstrap",
        type=int,
        default=None,
        metavar="N",
        help="Standard bootstrap replicates (slower, e.g. -b 100)"
    )
    boot_group.add_argument(
        "--alrt",
        type=int,
        default=None,
        metavar="N",
        help="SH-aLRT branch test replicates (e.g. --alrt 1000)"
    )

    # Tree options
    tree_group = parser.add_argument_group("Tree Options")
    tree_group.add_argument(
        "-o", "--outgroup",
        default=None,
        help="Outgroup sequence name for rooting (e.g. -o HXB2_reference)"
    )
    tree_group.add_argument(
        "-t", "--start-tree",
        default=None,
        help="Starting tree file (Newick format)"
    )
    tree_group.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent tree searches (default: 1)"
    )

    # Threads
    parser.add_argument(
        "-nt", "--threads",
        default="AUTO",
        help="Number of CPU threads. Use AUTO for automatic detection (default: AUTO)"
    )

    # Output
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output prefix for all result files (default: input filename)"
    )
    parser.add_argument(
        "--outdir",
        default="iqtree_results",
        help="Output directory for results (default: iqtree_results)"
    )

    # IQ-TREE path
    parser.add_argument(
        "--iqtree-path",
        default="iqtree2",
        help="Path to IQ-TREE2 executable (default: iqtree2)"
    )

    # Additional flags
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress IQ-TREE screen output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it"
    )

    return parser.parse_args()


# ── Validators ────────────────────────────────────────────────────────────────
def validate_inputs(args):
    """Validate input file and IQ-TREE installation."""

    # Check input file exists
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Check input file is not empty
    if os.path.getsize(args.input) == 0:
        logger.error(f"Input file is empty: {args.input}")
        sys.exit(1)

    # Check IQ-TREE is installed
    try:
        result = subprocess.run(
            [args.iqtree_path, "--version"],
            capture_output=True, text=True
        )
        version_line = result.stdout.split("\n")[0] if result.stdout else "Unknown"
        logger.info(f"IQ-TREE detected: {version_line.strip()}")
    except FileNotFoundError:
        logger.error(
            f"IQ-TREE2 not found at '{args.iqtree_path}'.\n"
            "Install with: conda install -c bioconda iqtree\n"
            "Or specify path with --iqtree-path"
        )
        sys.exit(1)

    # Warn if both bootstrap options specified
    if args.ultrafast_bootstrap and args.standard_bootstrap:
        logger.warning(
            "Both -bb and -b specified. Ultrafast bootstrap (-bb) will take priority."
        )


# ── Command Builder ───────────────────────────────────────────────────────────
def build_command(args, output_prefix):
    """Build the IQ-TREE command from parsed arguments."""

    cmd = [args.iqtree_path]

    # Input
    cmd += ["-s", args.input]

    # Model
    cmd += ["-m", args.model]
    if args.mset:
        cmd += ["--mset", args.mset]

    # Bootstrap — ultrafast takes priority
    if args.ultrafast_bootstrap:
        cmd += ["-bb", str(args.ultrafast_bootstrap)]
    elif args.standard_bootstrap:
        cmd += ["-b", str(args.standard_bootstrap)]

    # SH-aLRT
    if args.alrt:
        cmd += ["--alrt", str(args.alrt)]

    # Outgroup
    if args.outgroup:
        cmd += ["-o", args.outgroup]

    # Starting tree
    if args.start_tree:
        if not os.path.isfile(args.start_tree):
            logger.error(f"Starting tree file not found: {args.start_tree}")
            sys.exit(1)
        cmd += ["-t", args.start_tree]

    # Multiple runs
    if args.runs > 1:
        cmd += ["--runs", str(args.runs)]

    # Threads
    cmd += ["-nt", str(args.threads)]

    # Output prefix
    cmd += ["--prefix", output_prefix]

    # Flags
    if args.redo:
        cmd += ["--redo"]
    if args.quiet:
        cmd += ["--quiet"]

    return cmd


# ── Output Directory Setup ────────────────────────────────────────────────────
def setup_output(args):
    """Create output directory and determine output prefix."""

    os.makedirs(args.outdir, exist_ok=True)

    if args.prefix:
        prefix = args.prefix
    else:
        basename = os.path.splitext(os.path.basename(args.input))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{basename}_{timestamp}"

    output_prefix = os.path.join(args.outdir, prefix)
    return output_prefix


# ── Results Summary ───────────────────────────────────────────────────────────
def summarise_results(output_prefix):
    """Print a summary of output files generated by IQ-TREE."""

    extensions = {
        ".treefile":    "Best-fit phylogenetic tree (Newick format) — open in FigTree",
        ".iqtree":      "Full IQ-TREE report — model, log-likelihood, tree stats",
        ".log":         "Run log file",
        ".model.gz":    "ModelTest results (if -m TEST was used)",
        ".contree":     "Consensus tree from bootstrap replicates",
        ".ckp.gz":      "Checkpoint file (for resuming interrupted runs)",
        ".mldist":      "Maximum likelihood pairwise distance matrix",
        ".bionj":       "BioNJ starting tree",
    }

    logger.info("\n" + "="*60)
    logger.info("IQ-TREE RUN COMPLETE — OUTPUT FILES")
    logger.info("="*60)

    found_any = False
    for ext, description in extensions.items():
        filepath = output_prefix + ext
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            logger.info(f"  {ext:<20} {description}")
            logger.info(f"  {'':20} → {filepath} ({size:,} bytes)")
            found_any = True

    if not found_any:
        logger.warning("No output files found. Check the log for errors.")

    treefile = output_prefix + ".treefile"
    if os.path.isfile(treefile):
        logger.info("\n" + "="*60)
        logger.info("NEXT STEPS")
        logger.info("="*60)
        logger.info(f"  1. Open tree in FigTree:    {treefile}")
        logger.info(f"  2. Root on outgroup (if not already specified with -o)")
        logger.info(f"  3. Check bootstrap values — branches >70% are well supported")
        logger.info(f"  4. Full report:             {output_prefix}.iqtree")
    logger.info("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()

    logger.info("="*60)
    logger.info("IQ-TREE2 PYTHON WRAPPER")
    logger.info("="*60)
    logger.info(f"Input file  : {args.input}")
    logger.info(f"Model       : {args.model}")
    logger.info(f"Threads     : {args.threads}")
    logger.info(f"Output dir  : {args.outdir}")

    # Validate
    validate_inputs(args)

    # Setup output
    output_prefix = setup_output(args)
    logger.info(f"Output prefix: {output_prefix}")

    # Build command
    cmd = build_command(args, output_prefix)

    logger.info("\nIQ-TREE command:")
    logger.info("  " + " ".join(cmd) + "\n")

    # Dry run — print and exit
    if args.dry_run:
        logger.info("DRY RUN — command not executed.")
        sys.exit(0)

    # Run IQ-TREE
    logger.info("Running IQ-TREE2...\n")
    start_time = datetime.now()

    try:
        process = subprocess.run(
            cmd,
            text=True,
            check=True
        )
        end_time = datetime.now()
        elapsed = end_time - start_time
        logger.info(f"\nIQ-TREE completed successfully in {elapsed}")

    except subprocess.CalledProcessError as e:
        logger.error(f"IQ-TREE failed with return code {e.returncode}")
        logger.error("Check the .log file in your output directory for details.")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nRun interrupted by user. Use --redo to restart.")
        sys.exit(1)

    # Summarise results
    summarise_results(output_prefix)


if __name__ == "__main__":
    main()
