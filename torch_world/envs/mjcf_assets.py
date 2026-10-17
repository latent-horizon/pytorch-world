"""Compose packaged hardware models into our task scenes without runtime downloads."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET


@lru_cache
def _resource(path: str) -> bytes:
    return files("torch_world.envs").joinpath("assets", path).read_bytes()


@dataclass(frozen=True)
class MjcfAttachment:
    """A packaged MJCF model attached to a named scene body.

    Component names must be unique within the scene. Hardware options and named
    defaults are preserved; scene-wide integrator and timestep remain unchanged.
    """

    resource: str
    body: str

    def attach(self, scene: ET.Element) -> None:
        model = ET.fromstring(_resource(self.resource))
        mount = scene.find(f".//body[@name='{self.body}']")
        if mount is None:
            raise ValueError(f"Missing attachment body: {self.body}")
        mount.extend(model.find("worldbody"))
        directory = PurePosixPath(self.resource).parent
        meshdir = model.find("compiler").get("meshdir", "")
        for mesh in model.findall("asset/mesh"):
            mesh.set("file", str(directory / meshdir / mesh.get("file")))
        # Our scene has joint/geom defaults. Imported named classes must retain
        # MuJoCo's original defaults where the hardware model leaves them implicit.
        for default in model.findall("default/default"):
            for kind, attributes in (
                ("joint", {"damping": "0", "armature": "0"}),
                ("geom", {"density": "1000", "friction": "1 .005 .0001", "rgba": ".5 .5 .5 1"}),
            ):
                element = default.find(kind)
                if element is None:
                    element = ET.SubElement(default, kind)
                for key, value in attributes.items():
                    if key not in element.attrib:
                        element.set(key, value)
        for section in ("asset", "default", "contact", "tendon", "equality", "actuator"):
            source = model.find(section)
            if source is not None:
                destination = scene.find(section)
                if destination is None:
                    destination = ET.SubElement(scene, section)
                destination.extend(source)
        options = model.find("option")
        if options is not None:
            scene.find("option").attrib.update(
                {
                    key: value
                    for key, value in options.attrib.items()
                    if key not in {"timestep", "integrator", "gravity"}
                }
            )

    def assets(self) -> dict[str, bytes]:
        model = ET.fromstring(_resource(self.resource))
        directory = PurePosixPath(self.resource).parent / model.find("compiler").get("meshdir", "")
        paths = [str(directory / mesh.get("file")) for mesh in model.findall("asset/mesh")]
        return {path: _resource(path) for path in paths}
