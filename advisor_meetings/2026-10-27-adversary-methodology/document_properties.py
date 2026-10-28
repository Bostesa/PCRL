"""Set local document metadata only. This script has no research-data access."""
from pathlib import Path
import sys, zipfile
import xml.etree.ElementTree as ET
p = Path(sys.argv[1])
with zipfile.ZipFile(p) as z:
    members = [(i, z.read(i.filename)) for i in z.infolist()]
ns = {'dc': 'http://purl.org/dc/elements/1.1/', 'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties'}
ET.register_namespace('dc', ns['dc'])
ET.register_namespace('cp', ns['cp'])
values = {'dc:title':'Adversary and methodology agreement', 'dc:creator':'Nathan Samson', 'cp:lastModifiedBy':'Nathan Samson'}
with zipfile.ZipFile(p.with_suffix('.metadata.pptx'), 'w', zipfile.ZIP_DEFLATED) as out:
    for info, data in members:
        if info.filename == 'docProps/core.xml':
            root = ET.fromstring(data)
            for key, value in values.items():
                prefix, name = key.split(':')
                tag = f'{{{ns[prefix]}}}{name}'
                el = root.find(tag)
                if el is None:
                    el = ET.SubElement(root, tag)
                el.text = value
            data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        out.writestr(info, data)
p.with_suffix('.metadata.pptx').replace(p)
