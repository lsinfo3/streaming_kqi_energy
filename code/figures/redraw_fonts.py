"""Re-export the three data figures with embedded TrueType (Type 42) fonts instead of Type 3.
Same data, sizes and layout as the v5 scripts; only the font embedding and typeface change."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact_measurements as cm
import compact_scores as cs
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix"})
cm.offline(); cm.live(); cs.scores_fig()
print("saved offline_pairs, live_delta, test_scores")
