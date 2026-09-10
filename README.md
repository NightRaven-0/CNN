# Teaching a chest X-ray CNN to say *where*, and then checking whether it's telling the truth

**Project proposal — undergraduate thesis / capstone**
Dataset: VinDr-CXR · Hardware: single RTX 5070 Ti (16 GB) · Framework: PyTorch 2.11 / CUDA 12.8

---

## The short version

We're training a CNN on chest radiographs to do two things at once: name the abnormality, and name the part of the chest it's in. Then we compare it against the standard approach, which is to train a plain classifier and afterwards run Grad-CAM over it to guess what the model was looking at.

The question we want to answer is whether asking a model to localise *during training* produces better explanations than interrogating a black box *after* training, and what that costs in accuracy. VinDr-CXR ships with bounding boxes drawn by radiologists, so we can grade both approaches against a real answer key instead of squinting at heatmaps and deciding they look about right.

That last part is the whole point. Most papers in this area stop at "here is a heatmap, it looks plausible." We want a number.

## Why not just do the normal thing

The normal thing is: fine-tune a DenseNet or a ResNet on a public chest X-ray set, report AUROC per finding, generate Grad-CAM overlays, write "the model attends to clinically relevant regions," submit. There are hundreds of these. It's a fine exercise and a weak paper, and the reason it's weak isn't just saturation. The evidence says the heatmaps often aren't trustworthy:

- Arun and colleagues (*Radiology: AI*, 2021) scored saliency maps on chest X-ray localisation and found several popular methods lost to a dumb baseline that just draws a box where the finding usually is.
- Adebayo's sanity-check paper (NeurIPS 2018) showed that some saliency methods produce nearly the same picture after you randomise the model's weights, which means they're partly just edge detectors wearing a lab coat.
- DeGrave et al. (*Nature Machine Intelligence*, 2021) found COVID classifiers keying off laterality markers and image borders while the saliency maps still looked reasonable.

So "add explainability" as a final step is not a contribution. Building a model whose spatial reasoning is an output you can grade, and then actually grading it, is.

## What we settled on, and what we dropped

The first version of this proposal had a router: a gate that segments the chest into anatomical regions and dispatches each one to its own specialist CNN. Four experts, one for lung parenchyma, one for pleura, one for mediastinum, one for the skeleton. It's a nice idea and it matches the "localised, area-specific" framing we started from.

We're not doing it, at least not as the main contribution. Three reasons.

VinDr has no anatomy segmentations, so we'd have to bolt on a separate lung-field model and inherit its errors. The rare findings get worse, not better: rib fracture is well under 1% prevalence, and splitting an already-thin class across a dedicated head that sees almost nothing is a good way to learn nothing. And most importantly, we couldn't articulate why routing would improve classification accuracy over a backbone that already encodes position. It's a lot of machinery for an uncertain payoff, which is a bad trade when there's one person and one GPU.

What replaces it keeps the useful part. A single ConvNeXt-Tiny backbone with two heads:

- **Finding head** — 14 abnormality labels, multi-label.
- **Zone head** — which anatomical zone contains the finding. Six lung zones (left and right, upper/middle/lower) plus mediastinum and skeletal periphery.

The zone labels cost nothing extra. They fall out of the bounding boxes VinDr already provides. And the output reads the way a radiology report reads: *consolidation, right lower zone*, rather than a blurry orange blob.

The router isn't dead, it's just demoted to an optional third arm if the core work finishes early.

## Trade-offs, honestly

| Decision | What we get | What it costs |
|---|---|---|
| VinDr-CXR over CheXpert / NIH ChestX-ray14 | Radiologist-drawn boxes, and labels read by humans rather than scraped out of reports by an NLP tool | Smaller (18k vs 220k+), and access takes paperwork |
| ConvNeXt-Tiny backbone | Modern, ~28M params, trains overnight on 16 GB | Less directly comparable to the older DenseNet-121 literature, so we run that as a second baseline |
| Zone head instead of expert router | Same anatomical grounding, a tenth of the machinery, labels are free | Less novel-sounding on paper; we lose the "system of specialists" story |
| Zone classification instead of box regression | No detection framework to build or tune, weak supervision is enough | Coarser localisation. We can say "right lower zone", not "this 3 cm nodule" |
| 1024 px resized images | Fits on disk (~15–20 GB vs ~190 GB of DICOM), trains fast | Small nodules and subtle pneumothoraces may be lost in downsampling. This is a real limitation and we'll say so |
| Single dataset | Achievable | No cross-institution validation unless we add CheXlocalize later, which we'd like to |

## Where the novelty actually is

Three things, roughly in order of how confident we are about them.

**We're measuring explanations instead of displaying them.** Pointing game (does the model's peak fall inside the radiologist's box), IoU against those boxes, and zone-hit rate. Plus a deletion/insertion faithfulness test, and the Adebayo randomisation check on every method we report, including our own. Very few student projects in this space run that last one, and it's cheap.

**We're testing a specific claim rather than building a demo.** The claim is that supervised localisation beats post-hoc attribution on the same backbone with the same data, and that the accuracy penalty is small. It might turn out to be false. That's fine, and honestly a negative result here is more useful than another 0.94 AUROC.

**The explanation is in clinical vocabulary.** Anatomical zones are how findings get described in reports. A heatmap isn't. This is a smaller point but it's the one a clinician would care about.

## Scope, in three tiers

We are deliberately not trying to do everything.

**Core — this is the thesis.**
1. Baseline: DenseNet-121, 14-label classification. Get into the published ballpark so the numbers are credible.
2. Grad-CAM over that baseline, scored against boxes. Reproduces and extends the Arun result on VinDr.
3. Ours: ConvNeXt-T with the zone head. Same classification metrics, plus localisation.
4. Sanity check both.
5. Write it up, including whatever didn't work.

**If there's time.**
6. Per-zone specialist heads — the router idea, run as an ablation to see whether it earns its cost.
7. External validation on CheXlocalize, which has expert segmentations over CheXpert. Different hospital, different scanners. This is the single best test of whether the model learned anatomy or learned the dataset.

**Probably not.**
8. A Faster R-CNN / YOLO detection baseline. It's a whole project by itself.

## Things that could go wrong

*The zone head might not help.* Possible. If so, we report it, and the thesis becomes "we tested whether supervised localisation improves explanation quality on chest radiographs, and it didn't, and here's what that suggests." Still a result.

*The ground truth disagrees with itself.* Each VinDr image was read by 3 radiologists out of a pool of 17, and they don't always agree on where a finding is. We'll need a box-fusion rule and we should report inter-reader agreement, because the model can't beat the noise ceiling and knowing the ceiling stops us overclaiming.

*Rare findings may be unlearnable.* At 18k images, some classes have a few dozen positives. We're reporting per-class rather than hiding behind a macro average, and any class we can't evaluate gets excluded from the headline number and named in the text.

*Downsampling could hide the findings we most care about.* Small nodules especially. A high-resolution confirmation run on a subset is the mitigation.

## Reporting

We're filling in the CLAIM checklist (Checklist for Artificial Intelligence in Medical Imaging) as we go rather than at the end. It's what reviewers in this field expect and it forces us to write down decisions while we still remember making them.

## Repository

```
src/arc/
  zones.py      anatomical taxonomy, finding -> region mapping
  splits.py     patient-level stratified splitting + leakage checks
  data.py       (next) dataset, box fusion, zone label derivation
  models/       (next) backbone, heads, baselines
  evaluate.py   (next) classification, localisation, faithfulness
tests/          currently 8 tests, all on the splitter
paper/          manuscript, figures, CLAIM checklist
data/           gitignored
```

The splitter came first on purpose. Splitting on rows instead of on patients is the quiet mistake that invalidates a whole study, so it's written, tested, and asserted before every training run rather than trusted.
