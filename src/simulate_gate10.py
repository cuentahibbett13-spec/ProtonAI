import argparse
from pathlib import Path
import shutil


def _import_gate():
    try:
        import opengate as gate
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar opengate. Activa .venv e instala requirements-gate10.txt"
        ) from exc
    return gate


def copy_mhd_with_raw(source_mhd: str, target_mhd: str) -> None:
    src_mhd = Path(source_mhd)
    if not src_mhd.exists():
        raise FileNotFoundError(f"Density source MHD not found: {src_mhd}")

    lines = src_mhd.read_text(encoding="utf-8").splitlines()
    src_raw_name = None
    for line in lines:
        if line.strip().startswith("ElementDataFile") and "=" in line:
            src_raw_name = line.split("=", 1)[1].strip()
            break
    if src_raw_name is None:
        raise ValueError(f"ElementDataFile not found in {src_mhd}")

    src_raw = src_mhd.parent / src_raw_name
    if not src_raw.exists():
        raise FileNotFoundError(f"Density source RAW not found: {src_raw}")

    dst_mhd = Path(target_mhd)
    dst_mhd.parent.mkdir(parents=True, exist_ok=True)
    dst_raw = dst_mhd.with_suffix(".raw")

    new_lines = []
    for line in lines:
        if line.strip().startswith("ElementDataFile") and "=" in line:
            new_lines.append(f"ElementDataFile = {dst_raw.name}")
        else:
            new_lines.append(line)

    dst_mhd.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    shutil.copyfile(src_raw, dst_raw)


def resolve_dose_output(output_dose: str) -> None:
    requested_mhd = Path(output_dose)
    if requested_mhd.exists():
        return

    produced_mhd = requested_mhd.with_name(f"{requested_mhd.stem}_edep{requested_mhd.suffix}")
    if not produced_mhd.exists():
        raise FileNotFoundError(
            f"Dose output not found. Expected {requested_mhd} or {produced_mhd}"
        )

    copy_mhd_with_raw(str(produced_mhd), str(requested_mhd))


def run_simulation(
    phantom_mhd: str,
    labels_to_materials: str,
    material_db: str,
    output_dose: str,
    output_density: str,
    density_map_mhd: str,
    primaries: int,
) -> None:
    gate = _import_gate()
    u = gate.g4_units

    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False

    world = sim.world
    world.size = [1.0 * u.m, 1.0 * u.m, 1.0 * u.m]
    world.material = "G4_AIR"

    sim.volume_manager.add_material_database(material_db)

    phantom = sim.add_volume("Image", "phantom")
    phantom.image = phantom_mhd
    phantom.material = "G4_AIR"
    phantom.translation = [0.0, 0.0, 0.0]
    phantom.voxel_materials = [
        [0, 0, "G4_WATER"],
        [1, 1, "CORTICAL_BONE_1850"],
        [2, 2, "LUNG_0200"],
    ]

    source = sim.add_source("GenericSource", "protonBeam")
    source.particle = "proton"
    source.n = int(primaries)
    source.position.type = "disc"
    source.position.radius = 0.5 * u.mm
    source.position.translation = [0.0, 0.0, -90.0 * u.mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.energy.type = "mono"
    source.energy.mono = 150.0 * u.MeV

    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = "phantom"
    dose.size = [128, 128, 128]
    dose.spacing = [1.0 * u.mm, 1.0 * u.mm, 1.0 * u.mm]
    dose.output_filename = output_dose

    if labels_to_materials:
        phantom.voxel_materials = []
        with open(labels_to_materials, "r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                i_min, i_max, material = row.split()
                phantom.voxel_materials.append([int(i_min), int(i_max), material])

    sim.run()
    resolve_dose_output(output_dose)
    copy_mhd_with_raw(density_map_mhd, output_density)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GATE 10 simulation using Python API (opengate)")
    parser.add_argument("--phantom-mhd", type=str, default="data/phantom/sandwich_labels.mhd")
    parser.add_argument("--labels-to-materials", type=str, default="data/phantom/labels_to_materials.txt")
    parser.add_argument("--material-db", type=str, default="gate/materials/sandwich_materials.db")
    parser.add_argument("--output-dose", type=str, required=True)
    parser.add_argument("--output-density", type=str, default="data/gate/density_map.mhd")
    parser.add_argument("--density-map-mhd", type=str, default="data/phantom/sandwich_density.mhd")
    parser.add_argument("--primaries", type=int, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    Path(args.output_dose).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_density).parent.mkdir(parents=True, exist_ok=True)

    run_simulation(
        phantom_mhd=args.phantom_mhd,
        labels_to_materials=args.labels_to_materials,
        material_db=args.material_db,
        output_dose=args.output_dose,
        output_density=args.output_density,
        density_map_mhd=args.density_map_mhd,
        primaries=args.primaries,
    )

    print(f"Simulation complete: dose={args.output_dose}, density={args.output_density}")


if __name__ == "__main__":
    main()
