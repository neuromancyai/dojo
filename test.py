from pathlib import Path

from mujoco import MjModel

from dojo.brax import Environment


def main():
    model_file = Path("./scene.xml")
    mj_model = MjModel.from_xml_string(model_file.read_text())
    env = Environment(mj_model)


if __name__ == "__main__":
    main()
