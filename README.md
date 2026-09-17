# Explainable CNN for Chest X-ray Diagnosis: Project Brief

Prepared by: [team names]
For: [mentor name]
Status: trained and evaluated, results below

## What this is

This is the model and experiments we built for our mentor's paper, and the numbers they produced. Our part was the working system: a trained classifier, an explanation layer on top of it, and results that sit beside published work. The writing, the framing and the choice of venue belong to our mentor. Anything below can change, and some parts are already marked as "later".

The priorities, in order:

1. **Explainability.** For every prediction the system shows which part of the X-ray drove it. We also measure how far those explanations can be trusted, instead of only displaying them.
2. **Localisation.** The same explanation maps are turned into boxes and anatomical zones ("right lower zone") and scored against boxes drawn by radiologists.

## Dataset: NIH ChestX-ray14

112,120 frontal chest X-rays from 30,805 patients, released by the NIH Clinical Center. Each image has up to 14 disease labels, or "No Finding". The images are 1024×1024 PNG files, about 45 GB in total.

We picked it for three reasons.

It comes with an official patient-wise split: 86,524 images for training and validation, 25,596 for testing. Using that split is the only way our numbers can be compared directly with published ones.

The release also includes 984 hand-drawn bounding boxes on 880 images, covering 8 of the 14 classes, and every one of those images is in the official test list. That's small. Still, since the boxes never touch training, they give us an independent answer key for grading the explanations.

And we can start today. NIH places no restrictions on use. The only conditions are attribution: link to the NIH download page, cite Wang et al. (CVPR 2017), and acknowledge the NIH Clinical Center as the data provider. There's no account or data-use agreement involved.

The downsides are real. The labels were extracted from radiology reports by a text-mining tool, so some of them are wrong, and every paper on this dataset inherits that problem. The boxes only cover 8 classes and about a thousand images, so explanation scores can only be computed for those. Every image comes from the same hospital.

Our first plan used VinDr-CXR, which has cleaner labels and many more boxes. We moved away from it for now because few published classification results exist on a fixed VinDr split, so there'd be little to compare against, and getting access means signing agreements through Kaggle or PhysioNet. It stays on the list as a second test set.

## Model

DenseNet-121 pretrained on ImageNet, with 14 sigmoid outputs, one per disease. CheXNet used this architecture, and it's the most common baseline on this dataset, so the numbers are easy to place.

One change from CheXNet: we train on 512×512 images instead of 224×224. The explanation maps come from the last convolutional layer, which is a 7×7 grid at 224 and a 16×16 grid at 512. A 7×7 grid is too coarse to point at a nodule, and the results show that even 16×16 is too coarse for box-overlap scores to judge nodules fairly. The larger size costs about five times the compute per image, and our GPU (RTX 5070 Ti, 16 GB) can handle it.

Class imbalance is handled with a weighted loss. Hernia, for example, appears in well under 1% of images.

## Explanation layer

This is the main part of the project. Four explanation methods were evaluated on the trained model, next to a random control:

- CAM, the original class activation map. For this architecture it adds up exactly to the model's prediction, so it is a decomposition of the answer rather than an estimate of it, and it costs nothing extra to compute.
- Grad-CAM and Grad-CAM++, gradient-weighted activation maps and the usual choice in medical imaging papers
- EigenCAM, which takes the main component of the last layer's activations. It uses no gradients and ignores which finding is being explained, so it shows the same region whatever the question.
- A random map, as a control. Any score a random map also earns is not evidence of anything, and we would rather find that out ourselves than have a reviewer find it.

Three more methods are in the code but have not been evaluated: Score-CAM, Integrated Gradients and occlusion. Integrated Gradients and occlusion have unit tests. Score-CAM needs one forward pass per channel, 1024 of them for this network, and the other two work pixel by pixel instead of on the 16 by 16 grid, so all three need a slower evaluation path. They are listed under later work, not reported as results.

The methods rest on different assumptions, which is why we run several. If they agree, that agreement counts as evidence. If they disagree, that's reported as well.

Each method is scored three ways.

*Does it point at the right place?* We use the pointing game (does the hottest spot on the map fall inside a radiologist's box) and the overlap (IoU) between the map and the box.

*Is it faithful to the model?* We use deletion and insertion curves (Petsiuk et al., 2018). Remove the pixels the map ranks as most important and see how quickly the prediction falls. A good ranking should make it fall fast.

*Does it depend on the model at all?* We use the model-randomisation test from Adebayo et al. (2018). Randomise the network's weights layer by layer and check whether the maps change. If they stay the same, the method was tracing edges in the image and wasn't explaining the model.

We do all this because heatmaps in medical imaging are often less reliable than they look. Arun et al. (2021) tested eight saliency methods on chest X-ray localisation. All eight failed at least one of their trust tests, and all eight localised worse than models trained specifically for it (a U-Net and a RetinaNet). DeGrave et al. (2021) showed COVID-19 classifiers leaning on shortcuts such as text markers and patient positioning rather than on the lungs. A heatmap on its own proves very little. What we're adding is the measurement.

## Localisation

This is the second priority, and most of it comes out of the explanation layer. Each heatmap is thresholded into a box and compared with the radiologist boxes using the measure Wang et al. used themselves, T(IoBB), at thresholds of 0.1, 0.25 and 0.5. Their numbers are our baseline. On this dataset, at the loosest threshold, their accuracy ran from 1.00 for cardiomegaly down to 0.15 for nodules.

The prediction script also names an approximate zone for the hottest part of a map: patient left or right, and upper, middle or lower. It comes from image geometry, not a lung segmentation, and has not been checked against anatomy, so it is a readout for demonstrations rather than a result.

## What "comparable results" means

Published mean AUROC on the official test split. Each figure was checked against the table in the paper itself:

| Work | Network and input | Mean AUROC |
|---|---|---|
| Wang et al. 2017 | ResNet-50, 1024 px | 0.745 |
| Yao et al. 2018 | ResNet and DenseNet, 512 px, no pretraining | 0.761 |
| Baltruschat et al. 2019 | ResNet-38, 448 px, plus age, sex and view position | 0.806 |
| Guendel et al. 2018 | DenseNet-121, 1024 px | 0.807 |
| Taslimi et al. 2022 (SwinCheX) | Swin-L transformer, 224 px | 0.810 |
| Guan and Huang 2020 | DenseNet-121 with category-wise attention | 0.816 |
| Ma et al. 2019 | two DenseNet-121s with cross-attention | 0.817 |
| **this project, three runs** | **DenseNet-121, 512 px** | **0.816 to 0.819** |
| Goel et al. 2024 | DenseNet-121, 224 px | 0.8202 |
| Hermoza et al. 2020 | DenseNet-121, 512 px | 0.821 |
| Kim et al. 2021 (XProtoNet) | DenseNet-121, 512 px | 0.822 |
| Goel et al. 2024 | DenseNet-121 with low-rank features, 224 px | 0.8235 |
| Xiao et al. 2023 | ViT-B pretrained on 0.5M chest X-rays, 224 px | 0.830 |

We measured 0.8158, and repeating the whole procedure with two more seeds gave 0.8190 and 0.8164, a mean of 0.8171. DenseNet-121 results on this split run from 0.807 to 0.822, and ours sit inside that range, a little below its top: Hermoza et al. and XProtoNet, both DenseNet-121 at 512 px like ours, report 0.821 and 0.822. Transformers pretrained on hundreds of thousands of chest X-rays reach 0.830. The result is in line with published work, not an improvement on it.

There's one trap here. CheXNet's often-quoted 0.841 was measured on its own random 70/10/20 split, not on the official one, and so were Yao et al.'s 2017 results; their official-split figure is the 2018 one above. Baltruschat et al. showed that the choice of split alone shifts results noticeably. Random-split numbers shouldn't go in the same table as official-split results unless the difference is stated.

## Results

Run on 12 September 2026. DenseNet-121 at 512 px, batch 32, early stopping at epoch 7 with the best weights from epoch 4. Validation mean AUROC 0.8533, test mean AUROC 0.8158 across all 25,596 official test images.

### Classification

Mean AUROC 0.8158, mean AUPRC 0.2878. Per-class figures with 95% bootstrap intervals are in `runs/densenet121_512/test_metrics_per_class.csv`. The distance between the two measures is the part worth reading:

| finding | prevalence | AUROC | AUPRC |
|---|---|---|---|
| Effusion | 18.2% | 0.830 | 0.511 |
| Emphysema | 4.3% | 0.937 | 0.451 |
| Consolidation | 7.1% | 0.741 | 0.160 |
| Fibrosis | 1.7% | 0.841 | 0.113 |
| Pneumonia | 2.2% | 0.711 | 0.048 |

Pneumonia scores 0.711 by AUROC and 0.048 by AUPRC. At that prevalence the model is close to useless for it in practice, and a table carrying only AUROC would hide the fact entirely.

The probabilities themselves are not calibrated. Across the eight findings that have boxes, the median true positive scores between 0.24 (Pneumonia) and 0.67 (Pneumothorax), so 0.5 is not a meaningful cut-off, and anyone using the model would need a threshold chosen per finding. Ranking is what it does well: for each finding, the median boxed test case sits between the 75th and 97th percentile of all test images, though individual cases vary widely.

### Run to run variation

The whole procedure was repeated three times with different seeds. A seed changes the weight initialisation, the shuffling, the augmentation and which 10% of the training data is held out for validation. The test set is fixed by NIH's official list and never chooses anything.

| run | seed | epochs | best epoch | best validation | test AUROC |
|---|---|---|---|---|---|
| `densenet121_512` | 1337 | 7 | 4 | 0.8533 | 0.8158 |
| `densenet121_512_seed2024` | 2024 | 8 | 5 | 0.8479 | 0.8190 |
| `densenet121_512_seed7` | 7 | 8 | 5 | 0.8483 | 0.8164 |

Mean 0.8171, spread 0.0032 between lowest and highest. All three stopped early after 7 or 8 epochs and kept weights from epoch 4 or 5, so the training behaviour is consistent.

Two details are worth keeping. The run with the best validation score, seed 1337, has the lowest test score of the three, so validation ranking does not predict test ranking across runs. And each run validates on a different split, so those validation numbers are not strictly comparable to each other in the first place. Within a run, validation is what picks the epoch to keep. Across runs nothing picks a winner, and the honest number is the mean with its spread.

All three runs are kept in full, with weights, per-epoch history and test metrics, under `runs/`. Everything else in this document comes from the first run, which was chosen before the other two existed. Picking the best of the three by test score would be choosing on the test set, which is the one thing the official split exists to prevent.

### Do the explanations point where the radiologists did

Scored over the 984 image and finding pairs that carry hand-drawn boxes. All of them fall in the test split, so no box was seen during training.

| method | pointing game | IoU | T(IoBB) 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|
| Grad-CAM++ | 0.544 | 0.212 | 0.685 | 0.550 | 0.376 |
| Grad-CAM | 0.496 | 0.182 | 0.617 | 0.500 | 0.337 |
| CAM | 0.493 | 0.176 | 0.607 | 0.498 | 0.336 |
| EigenCAM | 0.479 | 0.190 | 0.629 | 0.504 | 0.319 |
| random control | 0.228 | 0.091 | 0.371 | 0.261 | 0.131 |

Every method roughly doubles the random control, so the measure is not one that anything can pass.

The spread between findings is far wider than the spread between methods:

| finding | CAM pointing | CAM T(IoBB) 0.5 | random pointing |
|---|---|---|---|
| Cardiomegaly | 0.870 | 0.815 | 0.349 |
| Nodule | 0.203 | 0.000 | 0.114 |

Cardiomegaly, which is large and sits in the same place every time, is localised well by everything. Pneumothorax is the clearest failure. With Grad-CAM++, the best localiser overall, its pointing score is 0.22, the lowest of any finding for that method: most of its maps point somewhere other than the radiologist's box.

Nodules score almost as low, but for them the scores are the problem. The median nodule box is 70 by 68 pixels, about one cell of the 16 by 16 grid the explanations are computed on, while the next smallest finding, masses, has boxes three times that area. A box drawn from any map ends up around twelve times the size of the nodule, which caps IoU and T(IoBB) however well the map is placed. For all four methods, in every one of the 79 nodule cases, no placement of that box could have reached T(IoBB) 0.5, so the zero in that column says nothing about the explanations.

So we added a check the grid can answer: does the box drawn from the map contain the nodule's centre? It does in 61% of cases for Grad-CAM++, 59% for CAM and 57% for Grad-CAM, against 13% for a random map. EigenCAM, which ignores which finding it is explaining, manages 37%. The explanations find nodules far more often than chance, and the standard box scores cannot show it at this resolution. The numbers are in `runs/xai/nodule_resolution.csv`.

At the strictest threshold, Wang et al.'s own localisation table for this dataset has the same extremes, cardiomegaly highest and nodules lowest, and the same caveat may apply to it.

### Against published localisation numbers

Two published results use the same boxes. Wang et al. count a box as correct when at least half of it lies inside the radiologist's box (IoBB of 0.5 or more). Hermoza et al. use overlap instead (IoU of 0.3 or more), which also punishes a box of the wrong size. The table gives the share of boxes counted correct, with our Grad-CAM++ maps.

| measure and source | Atelectasis | Cardiomegaly | Effusion | Infiltration | Mass | Nodule | Pneumonia | Pneumothorax | mean |
|---|---|---|---|---|---|---|---|---|---|
| IoBB, Wang et al. 2017 | 0.19 | 0.95 | 0.42 | 0.65 | 0.31 | 0.00 | 0.48 | 0.27 | 0.41 |
| IoBB, ours | 0.10 | 0.83 | 0.39 | 0.58 | 0.20 | 0.00 | 0.56 | 0.16 | 0.35 |
| IoU, Wang et al. 2017 (as reported by Hermoza et al.) | 0.24 | 0.46 | 0.30 | 0.28 | 0.15 | 0.04 | 0.17 | 0.13 | 0.22 |
| IoU, Hermoza et al. 2020 | 0.37 | 0.99 | 0.37 | 0.54 | 0.35 | 0.04 | 0.60 | 0.21 | 0.43 |
| IoU, ours | 0.19 | 0.16 | 0.42 | 0.46 | 0.27 | 0.01 | 0.56 | 0.13 | 0.28 |

On Wang et al.'s own measure we are a little below the original baseline, with the same easy and hard findings. On IoU, Grad-CAM++ beats that baseline but sits well below Hermoza et al., whose method is built to localise. Most of that gap is cardiomegaly. Our box is drawn around the hottest 5% of the map, far smaller than an enlarged heart, so the overlap stays low even when the box sits inside the radiologist's. Neither comparison is exactly like for like: Wang et al. can place several boxes on one image and count the extra ones as false positives, while we place one. Our IoU figures come from the per-image scores in `runs/xai/xai_per_image.csv`.

### Are the explanations faithful to the model

Deletion blanks the pixels a map ranks highest and watches the score fall, so low is good. Insertion adds them to an empty image, so high is good. Fifty images per method.

| method | deletion | insertion | difference |
|---|---|---|---|
| Grad-CAM | 0.194 | 0.624 | 0.431 |
| CAM | 0.198 | 0.628 | 0.430 |
| Grad-CAM++ | 0.212 | 0.603 | 0.391 |
| EigenCAM | 0.264 | 0.570 | 0.306 |
| random control | 0.338 | 0.371 | 0.033 |

### Do they depend on the model at all

Randomise the network from the output backwards and watch whether the map moves. Mean absolute rank correlation over the last 20 of 242 layers, where close to zero is the answer we want.

| method | mean absolute correlation |
|---|---|
| CAM | 0.009 |
| Grad-CAM | 0.052 |
| EigenCAM | 0.152 |
| Grad-CAM++ | 0.199 |

All four pass, in that the correlation collapses instead of holding near one.

### The three rankings disagree

Grad-CAM++ localises best and has the weakest sanity result. Grad-CAM and plain CAM are the most faithful and the least dependent on the image, while localising a little worse. EigenCAM loses on two of the three and its SVD failed to converge under randomisation, which we had to handle rather than repair.

No method wins on everything. A paper quoting one number for one method is therefore picking its own answer, and that is the argument for measuring all three properties instead of one.

## Decisions and trade-offs

| Decision | What we gain | What it costs |
|---|---|---|
| NIH ChestX-ray14 over VinDr-CXR | Official split, many published baselines, no access paperwork | Noisier labels, far fewer boxes |
| DenseNet-121 | Direct comparison with CheXNet-style work | Not the strongest backbone around today |
| 512 px input | 16×16 explanation maps instead of 7×7 | About 5 times the compute per image, and a departure from CheXNet's 224 |
| Four explanation methods plus a random control | Agreement between methods becomes evidence, and one weak method can't sink the result | More compute, and more to explain in the paper |
| Post-hoc explanations first | Works on any trained model, and matches most published work | The explanation is worked out afterwards, not built into the model |
| Official split only | Our numbers line up with the literature | No freedom to pick a friendlier split, which is the whole reason for using it |

## How this differs from a typical student project

Most projects like this train a classifier and add a few Grad-CAM pictures at the end. We score the explanations against radiologist boxes, test whether they're faithful to the model, and run the randomisation check that most papers skip. The classification numbers are on the official split, so they can sit next to published work without the CheXNet split problem. When an explanation method fails, we say so and show examples.

## Known limitations

The labels are text-mined and some are wrong. We can't fix that, only state it.

Explanation scores exist only for the 8 classes that have boxes.

The explanation maps are 16 by 16, coarser than a nodule, so box-overlap scores cannot judge localisation for findings that small. The nodule results use a containment check instead, which we defined for this purpose and which has not been validated anywhere else.

All the data is from one hospital, so we have no evidence yet that the model works elsewhere. Zech et al. (2018) showed that chest X-ray models can lose a lot of accuracy at a new hospital.

A heatmap shows where the model looked. It doesn't show whether the model reasoned the way a radiologist would. None of this is clinical validation, and the system isn't meant for diagnosis.

## Later, after the first results

- Evaluating Score-CAM, Integrated Gradients and occlusion, which are written but not yet scored
- Calibrating the probabilities on the validation set, for example with temperature scaling, and choosing a threshold for each finding
- Explanations built into the model itself, either with an attention-pooling head or with a prototype network that says "this region looks like these training cases"
- A test set from a second hospital, such as VinDr-CXR or CheXlocalize
- A newer backbone such as ConvNeXt, and higher resolution
- A radiologist rating a sample of the explanations

## Questions for our mentor

1. Which venue or format do you have in mind? That decides how many extra experiments are worth running.
2. Should the first version include a test set from a second hospital, or should that wait for a follow-up?
3. Is 512 px fine, or should we also report a 224 px run so the comparison with CheXNet is exact?
4. Does the department need any ethics paperwork, even for public, de-identified data?

## Where the code stands

Written and tested (40 tests passing), and run end to end:

- data splitting by patient, with a check that fails loudly if a patient ever lands in two splits
- loading of the NIH labels, the official split and the boxes, checked against the real files
- the DenseNet-121 model, whose class activation maps add up exactly to its predictions
- seven explanation methods in code, four of them evaluated, and the scoring that grades them against the boxes
- scripts that download, preprocess, train, and produce the three results tables: classification scores, localisation against the boxes, and faithfulness alongside the sanity check
- a script that draws the review figures, a check on how far box scores can judge nodules at this resolution, a comparison across repeated runs, and a tool that runs the model on a single X-ray and saves its explanation

Measured on this machine: training at batch size 32 on 512 px images used about 12 GB of the GPU's 16 GB. The first epoch took 10.4 minutes while cuDNN tuned its kernels and the rest about 7.5 each, and early stopping ended the run after 7 epochs, about 55 minutes in all. Training saves a resumable checkpoint after every epoch. That was added after three power cuts during the data download, and was tested by interrupting a run on purpose.

## Reproducing the results

Everything runs from the project root. The environment is defined in `pyproject.toml`, and `uv` builds it with the PyTorch build this GPU needs.

```bash
uv sync --extra dev
uv run python scripts/download_nih.py
uv run python scripts/preprocess_nih.py
uv run python scripts/train.py
uv run python scripts/evaluate.py --run runs/densenet121_512
uv run python scripts/evaluate_xai.py --checkpoint runs/densenet121_512/best.pt
uv run python scripts/evaluate_faithfulness.py --checkpoint runs/densenet121_512/best.pt --randomisation-samples 8
uv run python scripts/check_nodule_resolution.py
uv run python scripts/make_figures.py
```

To measure how much the result moves when the procedure is repeated:

```bash
uv run python scripts/train.py --seed 2024 --out runs/densenet121_512_seed2024
uv run python scripts/evaluate.py --run runs/densenet121_512_seed2024
uv run python scripts/train.py --seed 7 --out runs/densenet121_512_seed7
uv run python scripts/evaluate.py --run runs/densenet121_512_seed7
uv run python scripts/compare_runs.py
```

The download is 45 GB and took about nine hours on a 1.4 MB/s connection; it picks up where it stopped if interrupted. Preprocessing takes about five minutes, training about an hour, and the evaluations another half hour.

To run the model on one X-ray and see its explanation:

```bash
uv run python scripts/predict.py data/processed/nih512/00021181_002.png
uv run python scripts/predict.py data/processed/nih512/00025662_006.png --finding Nodule --method gradcam++
```

It prints all 14 findings ranked by probability and saves a picture to `runs/predictions/`, with the radiologist's box drawn when the image has one. It warns if the image was used in training.

## References

Please check each one against the original before it goes into the paper.

- Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. NeurIPS 2018.
- Arun N, et al. Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. Radiology: Artificial Intelligence, 2021.
- Baltruschat IM, Nickisch H, Grass M, Knopp T, Saalbach A. Comparison of deep learning approaches for multi-label chest X-ray classification. Scientific Reports, 2019.
- Chattopadhay A, et al. Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. WACV 2018.
- DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 2021.
- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. CVPR 2017.
- Goel R, Nath U, Wang Y, Silva AC, Wu T, Yang Y. Learning low-rank feature for thorax disease classification. arXiv:2404.18933, 2024.
- Guan Q, Huang Y. Multi-label chest X-ray image classification via category-wise residual attention learning. Pattern Recognition Letters, 2020.
- Guendel S, et al. Learning to recognize abnormalities in chest X-rays with location-aware dense networks. arXiv:1803.04565, 2018.
- Hermoza R, Maicas G, Nascimento JC, Carneiro G. Region proposals for saliency map refinement for weakly-supervised disease localisation and classification. MICCAI 2020.
- Kim E, Kim S, Seo M, Yoon S. XProtoNet: Diagnosis in chest radiography with global and local explanations. CVPR 2021.
- Ma C, Wang H, Hoi SCH. Multi-label thoracic disease image classification with cross-attention networks. MICCAI 2019.
- Petsiuk V, Das A, Saenko K. RISE: Randomized input sampling for explanation of black-box models. BMVC 2018.
- Rajpurkar P, et al. CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225, 2017.
- Selvaraju RR, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017.
- Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. ICML 2017.
- Wang H, et al. Score-CAM: Score-weighted visual explanations for convolutional neural networks. CVPR Workshops 2020.
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. CVPR 2017.
- Taslimi S, Taslimi S, Fathi N, Salehi M, Rohban MH. SwinCheX: Multi-label classification on chest X-ray images with transformers. arXiv:2206.04246, 2022.
- Xiao J, Bai Y, Yuille A, Zhou Z. Delving into masked autoencoders for multi-label thorax disease classification. WACV 2023.
- Yao L, et al. Learning to diagnose from scratch by exploiting dependencies among labels. arXiv:1710.10501, 2017.
- Yao L, Prosky J, Poblenz E, Covington B, Lyman K. Weakly supervised medical diagnosis and localization from multiple resolutions. arXiv:1803.07703, 2018.
- Zech JR, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLOS Medicine, 2018.
- Zeiler MD, Fergus R. Visualizing and understanding convolutional networks. ECCV 2014.
