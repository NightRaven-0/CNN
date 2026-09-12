# Explainable CNN for Chest X-ray Diagnosis: Project Brief

Prepared by: [team names]
For: [mentor name]
Status: plan settled, data download next

## What this is

This is the plan for the model and experiments we're building for our mentor's paper. Our part is the working system: a trained classifier, an explanation layer on top of it, and a set of results that can sit next to published work. The writing, the framing and the choice of venue belong to our mentor. Anything below can change once the first results are in, and some parts are already marked as "later".

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

DenseNet-121 pretrained on ImageNet, with 14 sigmoid outputs, one per disease. CheXNet used this architecture, and it's the most common baseline on this dataset, so our numbers will be easy to place.

One change from CheXNet: we train on 512×512 images instead of 224×224. The explanation maps come from the last convolutional layer, which is a 7×7 grid at 224 and a 16×16 grid at 512. A 7×7 grid is too coarse to point at a nodule. The larger size costs about five times the compute per image, and our GPU (RTX 5070 Ti, 16 GB) can handle it.

Class imbalance is handled with a weighted loss. Hernia, for example, appears in well under 1% of images.

## Explanation layer

This is the main part of the project. We run six explanation methods on the same trained model:

- CAM, the original class activation map. For this architecture it adds up exactly to the model's prediction, so it is a decomposition of the answer rather than an estimate of it, and it costs nothing extra to compute.
- Grad-CAM and Grad-CAM++, gradient-weighted activation maps and the usual choice in medical imaging papers
- Score-CAM, the same idea without gradients. It needs one pass per channel and the last block has 1024 of them, so it runs on a subsample rather than the whole test set.
- Integrated Gradients, a pixel-level method that adds up gradients along a path from a blank image to the real one
- Occlusion, which hides patches of the image and watches how the prediction changes
- A random map, as a control. Any score a random map also earns is not evidence of anything, and we would rather find that out ourselves than have a reviewer find it.

The methods rest on different assumptions, which is why we run several. If they agree, that agreement counts as evidence. If they disagree, that's reported as well.

Each method is scored three ways.

*Does it point at the right place?* We use the pointing game (does the hottest spot on the map fall inside a radiologist's box) and the overlap (IoU) between the map and the box.

*Is it faithful to the model?* We use deletion and insertion curves (Petsiuk et al., 2018). Remove the pixels the map ranks as most important and see how quickly the prediction falls. A good ranking should make it fall fast.

*Does it depend on the model at all?* We use the model-randomisation test from Adebayo et al. (2018). Randomise the network's weights layer by layer and check whether the maps change. If they stay the same, the method was tracing edges in the image and wasn't explaining the model.

We do all this because heatmaps in medical imaging are often less reliable than they look. Arun et al. (2021) tested eight saliency methods on chest X-ray localisation. All eight failed at least one of their trust tests, and all eight localised worse than models trained specifically for it (a U-Net and a RetinaNet). DeGrave et al. (2021) showed COVID-19 classifiers leaning on shortcuts such as text markers and patient positioning rather than on the lungs. A heatmap on its own proves very little. What we're adding is the measurement.

## Localisation

This is the second priority, and most of it comes out of the explanation layer. Each heatmap is thresholded into a box and compared with the radiologist boxes using the measure Wang et al. used themselves, T(IoBB), at thresholds of 0.1, 0.25 and 0.5. Their numbers are our baseline. At the loosest threshold their accuracy ran from 0.99 for cardiomegaly down to 0.16 for nodules.

Boxes are also mapped onto six lung zones plus the mediastinum, so the output reads the way a report would.

## What "comparable results" means

Published mean AUROC on the official test split:

| Work | Mean AUROC | Notes |
|---|---|---|
| Wang et al. 2017 | 0.745 | original baseline, ResNet-50 |
| Yao et al. 2017 | 0.761 | |
| Baltruschat et al. 2019 | 0.806 | ResNet-38 using image plus patient age, sex and view position |
| arXiv:2404.18933 (2024) | 0.812 and 0.824 | DenseNet-121, baseline and improved version |

For a first working model we're aiming for a mean AUROC of about 0.80 to 0.82 on the official split. That's the same range as recent DenseNet-121 results, which is a fair reading of "comparable".

There's one trap here. CheXNet's often-quoted 0.841 was measured on its own random 70/10/20 split, not on the official one. Baltruschat et al. showed that the choice of split alone shifts results noticeably. That number shouldn't go in the same table as official-split results unless the difference is stated.

## Decisions and trade-offs

| Decision | What we gain | What it costs |
|---|---|---|
| NIH ChestX-ray14 over VinDr-CXR | Official split, many published baselines, no access paperwork | Noisier labels, far fewer boxes |
| DenseNet-121 | Direct comparison with CheXNet-style work | Not the strongest backbone around today |
| 512 px input | 16×16 explanation maps instead of 7×7 | About 5 times the compute per image, and a departure from CheXNet's 224 |
| Six explanation methods | Agreement between methods becomes evidence, and one weak method can't sink the result | More compute, and more to explain in the paper |
| Post-hoc explanations first | Works on any trained model, and matches most published work | The explanation is worked out afterwards, not built into the model |
| Official split only | Our numbers line up with the literature | No freedom to pick a friendlier split, which is the whole reason for using it |

## How this differs from a typical student project

Most projects like this train a classifier and add a few Grad-CAM pictures at the end. We score the explanations against radiologist boxes, test whether they're faithful to the model, and run the randomisation check that most papers skip. The classification numbers are on the official split, so they can sit next to published work without the CheXNet split problem. When an explanation method fails, we say so and show examples.

## Known limitations

The labels are text-mined and some are wrong. We can't fix that, only state it.

Explanation scores exist only for the 8 classes that have boxes.

All the data is from one hospital, so we have no evidence yet that the model works elsewhere. Zech et al. (2018) showed that chest X-ray models can lose a lot of accuracy at a new hospital.

A heatmap shows where the model looked. It doesn't show whether the model reasoned the way a radiologist would. None of this is clinical validation, and the system isn't meant for diagnosis.

## Later, after the first results

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

Written and tested so far (35 tests passing):

- data splitting by patient, with a check that fails loudly if a patient ever lands in two splits
- loading of the NIH labels, the official split and the boxes, checked against the real files
- the DenseNet-121 model, whose class activation maps add up exactly to its predictions
- the six explanation methods, and the scoring that grades them against the boxes
- scripts that download, preprocess, train, and produce the three results tables: classification scores, localisation against the boxes, and faithfulness alongside the sanity check

Measured on this machine rather than estimated: training at batch size 32 on 512 px images uses about 11 GB of the GPU's 16 GB and runs at 165 images per second, which is roughly 8.5 minutes per epoch. At most 12 epochs with early stopping puts a full run under two hours. The data download is still going, and training starts once all 112,120 images are on disk.

## References

Please check each one against the original before it goes into the paper.

- Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. NeurIPS 2018.
- Arun N, et al. Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. Radiology: Artificial Intelligence, 2021.
- Baltruschat IM, Nickisch H, Grass M, Knopp T, Saalbach A. Comparison of deep learning approaches for multi-label chest X-ray classification. Scientific Reports, 2019.
- Chattopadhay A, et al. Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. WACV 2018.
- DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 2021.
- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. CVPR 2017.
- Learning low-rank feature for thorax disease classification. arXiv:2404.18933, 2024.
- Petsiuk V, Das A, Saenko K. RISE: Randomized input sampling for explanation of black-box models. BMVC 2018.
- Rajpurkar P, et al. CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225, 2017.
- Selvaraju RR, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017.
- Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. ICML 2017.
- Wang H, et al. Score-CAM: Score-weighted visual explanations for convolutional neural networks. CVPR Workshops 2020.
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. CVPR 2017.
- Yao L, et al. Learning to diagnose from scratch by exploiting dependencies among labels. arXiv, 2017.
- Zech JR, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLOS Medicine, 2018.
- Zeiler MD, Fergus R. Visualizing and understanding convolutional networks. ECCV 2014.
