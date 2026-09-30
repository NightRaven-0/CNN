# Capstone Phase 1: what to write on the two forms

Everything below comes from the project as it stands on 23 September 2026. Every number matches
`README.md` and the saved runs under `runs/`, so nothing here has to be defended twice.

Left blank on purpose, because only the team and the department can fill them: project group
number, register numbers, student names, supervisor name, signatures and dates.

## The name

**ARC-CXR**, for *audited radiograph classification*.

`arc` is already the Python package (`src/arc`) and `arc-cxr` the distribution name in
`pyproject.toml`, so the name costs nothing in the code. The word doing the work is "audited".
A project of this kind usually trains a classifier and prints a Grad-CAM picture at the end.
This one grades the picture: against boxes drawn by radiologists, against a random map, and
against the model itself.

Full title, for the title field on both sheets:

> ARC-CXR: Audited Explanations for Multi-Label Chest X-ray Classification and Weakly-Supervised
> Localisation

Short form if a field is too small: *ARC-CXR: audited explanations for chest X-ray classification*.

## Sheet 1: Project Work Area / Title Identification

**Project Area**

Artificial intelligence and deep learning for medical image analysis, with explainable AI (XAI)
as the focus.

**Proposed Project Title**

ARC-CXR: Audited Explanations for Multi-Label Chest X-ray Classification and Weakly-Supervised
Localisation

**Short Description**

Chest X-ray classifiers are usually shipped with a heat map as proof that the model looked in
the right place, and that picture is rarely checked. ARC-CXR trains a DenseNet-121 on the
112,120 images of NIH ChestX-ray14 at 512 by 512 pixels and then audits the explanations it
produces. Each map is scored against boxes drawn by radiologists, tested for faithfulness by
deleting and restoring the pixels it calls important, and put through a weight randomisation
check, with a random map as the control. The same maps also give a box and an anatomical zone
for each finding. All results use the dataset's official patient-wise test split, so they sit
directly beside published work.

## Sheet 2: Project Assessment Form (Capstone Project Phase-1)

**Program**: B. Tech AI. **Project Title**: as above.

**Domain**

Computer vision and medical image analysis, with explainable AI (XAI). Task: multi-label
classification and weakly supervised localisation of thoracic findings in chest radiographs.

**Details of the Laboratory facilities to be utilized (if any)**

None needed from the department at this stage. Training and evaluation run on the team's own
workstation: an NVIDIA RTX 5070 Ti with 16 GB, of which about 12 GB is used at batch size 32,
with Python 3.12 and PyTorch 2.11 on CUDA 12.8. The dataset needs about 45 GB of local disk, and
one full training run takes about an hour. If a department GPU with more memory becomes
available in Phase 2, it would go to the higher resolution and transformer runs listed under
future work. No wet lab, clinical facility or patient contact is involved.

**Scope of Interdisciplinary Project: Yes**

Computing and diagnostic radiology. The labels, the bounding boxes used as the answer key and
the anatomical vocabulary the system reports in ("effusion, left lower zone") are clinical, and
Phase 2 asks a radiologist to rate a sample of the explanations.

**Industry/External Project: No**

Self-contained academic work under our faculty supervisor, on a public, de-identified dataset
released by the NIH Clinical Center. No company, hospital or external funding is involved.

**Expected Outcome**

Tick **Conference**. The working system and its results are meant to go into a paper with our
supervisor, who chooses the venue; tick **Journal** instead if they prefer one. **Knowledge
Sharing** can be ticked as well, since the code, the saved runs and the review deck are written
to be handed over and rerun. Not a Product: this is a research prototype and is not a
diagnostic tool.

**Project details (abstract)**

Deep networks report strong results on chest radiographs, but they are rarely trusted in
practice, and one reason is that the saliency maps offered as explanations are seldom verified.
Arun et al. (2021) tested eight saliency methods on chest X-ray localisation and found that all
eight failed at least one trust test. ARC-CXR therefore treats the explanation, not the
classifier, as the object of study.

We train a DenseNet-121 initialised from ImageNet on NIH ChestX-ray14, which holds 112,120
frontal radiographs from 30,805 patients labelled with 14 findings, using the official
patient-wise split of 86,524 training and validation images and 25,596 test images. Input
resolution is 512 by 512 rather than the usual 224, which costs about five times the compute
per image and buys explanation maps on a 16 by 16 grid instead of 7 by 7. Class imbalance is
handled with a weighted loss.

Four explanation methods, CAM, Grad-CAM, Grad-CAM++ and EigenCAM, are then graded three ways
against a random control: whether they point where radiologists drew the 984 boxes, whether
they are faithful to the model under deletion and insertion, and whether they depend on the
model at all under cascading weight randomisation. Scores are calibrated on validation data and
a decision cut-off is chosen for each finding, so the system reports probabilities with a
defensible operating point rather than raw numbers.

The model reaches a mean AUROC of 0.817 across three repeated runs, inside the 0.812 to 0.822
range published for DenseNet-121 between 2021 and 2024. Grad-CAM++ localises best, its hottest
point landing in the radiologist's box 2.4 times as often as a random map's, but it is also the
least sensitive to the model being scrambled, which is reported rather than hidden. Phase 2
adds a second hospital's data, explanations built into the model, and a radiologist's review.

### Shorter abstract, if the box on the sheet is small

ARC-CXR trains a DenseNet-121 on NIH ChestX-ray14 at 512 by 512 pixels, on the official
patient-wise split, and audits the explanations it produces. Four saliency methods and a random
control are scored against boxes drawn by radiologists, tested for faithfulness by deletion and
insertion, and checked against weight randomisation, and the outputs are calibrated with a
decision cut-off for each finding. Mean AUROC is 0.817 across three runs, in line with
published DenseNet-121 results, and the best method points inside the radiologist's box 2.4
times as often as chance.

## If the panel asks

**Objectives**

1. Train a multi-label classifier on chest radiographs whose accuracy can be compared directly
   with published work, by using the dataset's official patient-wise split and nothing else.
2. Produce an explanation for every prediction, and measure it three ways: does it point where
   a radiologist did, is it faithful to the model, does it depend on the model at all.
3. Turn the same explanation into a box and an anatomical zone, and score that against the
   radiologist boxes as weakly supervised localisation.
4. Make the outputs usable: calibrated probabilities and a decision cut-off per finding, both
   fitted on validation data only.

**What is already done, at the end of Phase 1**

Mean AUROC 0.8158 on the official test set, repeated with three seeds for 0.816 to 0.819. Four
explanation methods scored against 984 boxes, with a random control at roughly half their
score. Deletion, insertion and the randomisation check run. Calibration error down from 0.117
to 0.011, and a cut-off per finding giving a mean F1 of 0.35 against 0.13 for flagging every
image. Four runs saved in full with weights and history, 44 tests passing, and a first review
deck in `presentation/`.

**What is still wrong**

Pneumothorax maps point away from the radiologist's box more often than not. The 16 by 16 grid
is coarser than a nodule, so box overlap scores cannot judge the smallest findings and we use a
containment check instead. Labels in this dataset are text mined and some are wrong. All the
data comes from one hospital, so there is no evidence yet that any of it transfers. The cut-offs
for rare findings rest on few validation cases; Hernia has 16.

**What Phase 2 is for**

A second hospital's test set (VinDr-CXR or CheXlocalize), explanations built into the model
rather than added afterwards, the three remaining methods scored, finer maps for small findings,
and a radiologist rating a sample of the explanations.
