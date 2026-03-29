# Dataset Documentation

Fill this document in as you collect and annotate your data.
This is a required submission deliverable.

---

## Data Sources

### Base Dataset
- **Name**: Construction Site Safety — Roboflow Universe
- **URL**: https://universe.roboflow.com/roboflow-100/construction-site-safety
- **License**: MIT / CC BY 4.0 (check on Roboflow before using)
- **Images used**: TBD
- **Why chosen**: Diverse indoor/outdoor construction environments, pre-annotated with PPE classes

### Custom Additions
- **Source 1**: [e.g. Google Image Search — search terms used]
- **Source 2**: [e.g. YouTube construction site footage — channels / videos]
- **Source 3**: [e.g. Own photos taken at a local site]
- **Number of custom images added**: TBD

---

## Dataset Statistics

| Split | Images | Annotated boxes |
|-------|--------|-----------------|
| Train | TBD    | TBD             |
| Val   | TBD    | TBD             |
| Test  | TBD    | TBD             |
| **Total** | **TBD** | **TBD**    |

### Class Distribution

| Class | Count | % of total |
|-------|-------|-----------|
| worker | TBD | TBD |
| hard-hat | TBD | TBD |
| safety-vest | TBD | TBD |
| no-hard-hat | TBD | TBD |
| no-safety-vest | TBD | TBD |
| safety-harness | TBD | TBD |

---

## Environment Diversity Checklist

- [ ] Outdoor — open construction lot
- [ ] Outdoor — scaffolding / elevated work
- [ ] Indoor — warehouse / building interior
- [ ] Overcast / diffuse lighting
- [ ] Bright direct sunlight
- [ ] Shadow-heavy scenes
- [ ] Artificial lighting (night / indoor)
- [ ] Multiple workers in frame
- [ ] Single worker close to camera
- [ ] Workers at distance (small in frame)
- [ ] Partially occluded workers

---

## Annotation Approach

- **Tool used**: Roboflow (online annotation UI) / LabelImg (local)
- **Format**: YOLO — one `.txt` file per image, one line per object:
  `class_id cx cy width height` (all values normalised 0–1)
- **Who annotated**: [your name]
- **Annotation rules**:
  - Draw box tightly around the person's body for `worker`
  - Draw box around the helmet only for `hard-hat`
  - Draw box around the vest area for `safety-vest`
  - Use `no-hard-hat` when a worker's head is clearly visible with no helmet
  - Skip ambiguous cases where PPE status cannot be determined

---

## Known Dataset Limitations

- Limited night-time imagery — model may underperform in low light
- Most images sourced from Western construction sites — may not generalise to all site types
- Harness class has fewer examples — lower confidence expected for this class
