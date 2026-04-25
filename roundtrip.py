"""
AC outer product, live.

Sparse A of shape [p, k] and dense C of shape [k, T] live on the GPU. Each
animation tick selects a column t of C, computes y = A @ C[:, t] in a compute
shader, and writes the result reshaped to [m, n] into the texture backing a
fastplotlib ImageGraphic. No GPU->CPU->GPU round-trip.
"""
import fastplotlib as fpl

from pathlib import Path
import masknmf

adapter = fpl.enumerate_adapters()[1]
print(adapter.info)
fpl.select_adapter(adapter)

parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"
session = "2024-07-18"

session_path = parent_path.joinpath(subject, session)

dmr_path = session_path.joinpath(f"demix.hdf5")
dmr = masknmf.DemixingResults.from_hdf5(dmr_path)
dmr.to("cuda")

AC = dmr.ac_array
pmd = dmr.pmd_array
T = AC.shape[0]

fig = fpl.Figure(
    shape=(1, 2),
    names=["pmd", "ac"],
    size=(1200, 700),
    canvas_kwargs={"max_fps": 999, "vsync": False},
)
pmd_image = fig["pmd"].add_image(pmd[0].cpu().numpy().squeeze(), cmap="viridis")
ac_image = fig["ac"].add_image(AC[0].cpu().numpy().squeeze(), cmap="viridis")

t = 0
def tick(figure):
    global t
    if t == T:
        t = 0

    pmd_image.data = pmd[t].cpu().numpy().squeeze()
    ac_image.data = AC[t].cpu().numpy().squeeze()
    t += 1


fig.add_animations(tick)
fig.imgui_show_fps = True
fig.show()
fpl.loop.run()
