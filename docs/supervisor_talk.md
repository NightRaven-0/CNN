# Talk for our supervisor: what we chose, why, and what we did differently

About eleven minutes at a normal pace. Section headings are for you, not to be read out.
Every number here matches `README.md` and the saved runs, so anything you say can be checked
on the spot.

---

## Opening

Before the numbers, here is the one line version.

We trained a chest X-ray classifier that matches what is published, and then spent most of our
effort testing whether its explanations deserve to be believed. The classifier is the part that
makes us comparable. The audit is the part that makes us different. That is why we are calling
it ARC-CXR, for audited radiograph classification.

## 1. Why we picked this problem

Nearly every medical imaging paper puts a Grad-CAM picture in the results and calls it
explainability. Almost nobody grades the picture.

Arun and colleagues, in Radiology: Artificial Intelligence in 2021, tested eight saliency
methods on chest X-ray localisation. All eight failed at least one of their trust tests, and all
eight localised worse than models trained to localise. The same year DeGrave and colleagues
showed in Nature Machine Intelligence that COVID classifiers were reading text markers and
patient position instead of lungs.

So the heat map is the weakest link in the chain, and it is the one part that everyone shows and
nobody measures. We decided to measure it.

## 2. The dataset, and what we gave up for it

NIH ChestX-ray14. 112,120 frontal chest X-rays from 30,805 patients, 14 findings, about 45 GB.

Three reasons.

It ships with an official split by patient: 86,524 images for training and validation, 25,596
for test, and no patient on both sides. If we use that split and nothing else, our number can be
placed next to a published number and mean something. If we invent our own split, it cannot.

It includes 984 boxes drawn by radiologists on 880 images, covering 8 of the 14 findings, and
every one of those images sits in the test half. The boxes never touch training, so they are an
independent answer key for grading explanations. That is what this project needs, and
very few public sets have it.

And we could start the same day. No data use agreement, no account, no gatekeeper. NIH asks for
a citation and nothing else.

We know what we gave up. The labels were text mined out of radiology reports, so some of them
are wrong, and every paper on this dataset carries that problem. The boxes cover 8 classes and
about a thousand images, so explanation scores exist only for those. It is one hospital, so we
have no evidence yet about anywhere else.

We did start on VinDr-CXR, which has cleaner labels and far more boxes. We moved off it because
almost nobody publishes on a fixed VinDr split, so we would have had nothing to sit beside, and
access needs agreements. It stays on the list as a second test set, which is the right way to
use it anyway.

## 3. The model, and the one place we broke with convention

DenseNet-121, pretrained on ImageNet, with 14 sigmoid outputs, one per finding.

We did not choose it because it is the strongest network available today. It is not. We chose it
because it is the one recent work on this dataset keeps using, from XProtoNet at CVPR 2021 to
Goel and colleagues in 2024, so our number lands in a range people recognise. Seven million
parameters, of which 14,350 are new: the final classifier.

There is a second reason, and it is architectural. This network ends in global average pooling
followed by a single linear layer. That means the class activation map adds up exactly to the
score the model produced. The explanation is not an estimate of the model's reasoning, it is a
decomposition of the actual output. Very few architectures hand you that for free, and since
explanation quality is what we are measuring, we took it.

The one place we broke with convention is input size. Almost everyone trains at 224 by 224. We
train at 512. The reason is the map, not the accuracy: the explanation comes off the last
convolutional layer, which is a 7 by 7 grid at 224 and 16 by 16 at 512. A 7 by 7 grid cannot
point at a nodule. It cost us about five times the compute per image, roughly an hour per run on
our own GPU.

And we tested that decision instead of asserting it. We trained the identical recipe at 224 with
the same seed. It scored 0.8025, against 0.8158 at 512. That gap is about four times the spread
we see between seeds, so the larger input is buying accuracy as well as a finer map.

Class imbalance is handled in the loss, weighted by how rare each finding is. Hernia turns up in
well under one percent of images.

## 4. The explanation layer, which is the actual project

We score four explanation methods against a random map as a control: CAM, Grad-CAM, Grad-CAM++
and EigenCAM.

The control matters more than it sounds. Any score a random map can also earn is not evidence of
anything, and we would rather find that out ourselves than have a reviewer find it.

Each method faces three questions.

Does it point where the radiologist pointed? The pointing game and box overlap, over those 984
boxes.

Is it faithful to the model? Deletion and insertion curves, from Petsiuk's RISE paper. Remove the
pixels the map calls important and watch how fast the prediction falls. A good ranking makes it
fall fast.

Does it depend on the model at all? The cascading randomisation test from Adebayo and colleagues
at NeurIPS 2018. Scramble the network layer by layer and watch the map. If the map barely
changes while the model is being destroyed, that method was tracing edges in the image and
explaining nothing. Most medical imaging papers skip this test. We ran it 242 times.

## 5. What came out

Classification. Mean AUROC 0.8158 on the official test set. We repeated the whole procedure with
three seeds and got 0.816, 0.819 and 0.816, so the honest number is 0.817 with a spread of
0.003. DenseNet-121 results published between 2021 and 2024 run from 0.812 to 0.822. We are
inside that band, a little below its top. XProtoNet, the closest match to our setup, reports
0.822. Transformers pretrained on half a million chest X-rays reach 0.830 to 0.834. So we are in
line with published work and not an improvement on it, and for a measurement paper that is the
right place to stand.

Explanations. Every method roughly doubles the random control. Grad-CAM++ lands its hottest
point inside the radiologist's box 54 percent of the time, against 23 percent for a random map,
which is 2.4 times. But no method wins all three tests. Grad-CAM++ localises best and changes
least when the model is scrambled, which is the worst possible result on the sanity check. CAM
and Grad-CAM localise slightly worse and depend most on the model. That tension is a finding. We
report it rather than quoting the half that flatters us.

The bigger effect is the finding, not the method. Cardiomegaly is localised 87 percent of the
time; it is large and always in the same place. Nodules, 20 percent. Pneumothorax is the real
failure: 22 percent even with our best method, so most of its maps point somewhere other than
the box.

Calibration, which we finished this week. The raw scores rank images well but they are not
probabilities, because the weighted loss pushes every finding up by a different amount. We fit
the correction on validation data only. Calibration error falls from 0.117 to 0.011 and AUROC
does not move. Then a decision cut-off per finding, again from validation: precision 0.32,
recall 0.46, F1 0.35 on average, against 0.13 for a system that flags every image. Accuracy is
0.87, which is worse than the 0.92 you get by always answering "no finding", and that is exactly
why we do not quote accuracy on this dataset.

## 6. What we did differently

This is the part I want your view on.

One. We graded the explanation instead of displaying it. Three tests, not one picture: the
radiologist's box, faithfulness to the model, and dependence on the model.

Two. We ran a random control. Most papers do not, so their reader cannot tell how much of
the score comes from the method and how much comes from the fact that chests are shaped like
chests. Ours sits in every table.

Three. We ran the sanity check most papers skip, and it gave us an uncomfortable answer
about the method that scores best. We kept the answer.

Four. We never touched the test set to make a decision. Official split only. Every cut-off
fitted on validation. The run we report everywhere was chosen before the other two existed. We
could have quoted 0.819 by picking the best of three seeds. We quote 0.8158 and the spread,
because picking the best of three on test is the one thing the official split exists to prevent.

Five. We checked whether our own metric could answer the question we were asking it. The
median nodule box is 70 by 68 pixels. One cell of our map is 64 by 64. A box drawn from any map
ends up about twelve times the size of the nodule, and we proved that in all 79 nodule cases no
placement of that box could have reached the standard threshold. A zero there measures the
ruler, not the model. So we defined a check the grid can answer: does the box contain the
nodule's centre? Grad-CAM++ does that 61 percent of the time, a random map 13 percent. We label
that as our own definition and not a standard, because it is not one yet.

Six. We verified every published number we compare against, in the paper's own table. That
caught an error in our own materials: we had Goel's DenseNet-121 at 0.812 when the paper says
0.8202. We fixed the number and deleted the claim it was propping up. We also dropped
ThoraX-PriorNet's 0.847 from the comparison, because it is measured on a random split, and
numbers from different splits are not comparable.

Seven. We built it as software, not a notebook. 44 tests, checkpoints that resume after a
power cut, four runs saved in full with weights and history, and every table in the report
regenerated by a script.

## 7. What broke on the way

Three worth your time.

The randomisation check came back as NaN for every method. Scrambling a layer's weights leaves
BatchNorm's running mean and variance untouched, because those are not weights, and the stale
statistics pushed activations to infinity across the network. We reset them too, and now 242 of
242 correlations come back finite.

Boxes drawn from Grad-CAM maps were covering the entire X-ray. Grad-CAM clips negative values to
zero, so on a mostly empty map the 95th percentile is zero, and "at or above the threshold"
selects every pixel in the image. That is the dangerous kind of bug. It does not crash, it just
produces plausible numbers that are wrong.

And batch size 48 ran out of GPU memory without raising an error at all, because on Windows the
driver spills into system RAM instead of failing. Training got twenty times slower. We
stayed at batch 32.

## 8. What is still wrong, in our own words

Pneumothorax maps point away from the radiologist's box more often than they point at it.

Pneumonia is close to useless in practice: AUPRC 0.048 at 2.2 percent prevalence, about twice
chance.

The cut-offs for rare findings rest on very few validation cases. Hernia has 16.

Everything comes from one hospital. Zech and colleagues showed in 2018 how much a chest X-ray
model can lose at a new one.

And a heat map shows where the model looked, not whether it reasoned the way a radiologist
would. None of this is clinical validation, and we do not describe it as a diagnostic tool.

## 9. Where it goes, and what we need from you

Phase 2, in order: a second hospital's test set, explanations built into the model rather than
added afterwards, the three methods we have written but not yet scored, finer maps for small
findings, and a radiologist rating a sample of the explanations.

Four questions for you.

Which venue or format do you have in mind? That decides how many more experiments are worth
running, so it is the one answer that changes our plan.

Should the first version include a second hospital, or does that wait for a follow-up?

We have the same recipe at 224 and at 512, 0.802 against 0.816. Should the paper report both, or
only 512?

And does the department need any ethics paperwork, even for public, de-identified data?

## Close

The classifier is finished and it sits where it should sit. The audit is the contribution, and
it is the part we would like your judgement on.

---

## If they ask

**"Why not a transformer, if those score higher?"**
The ones that reach 0.830 to 0.834 are pretrained on hundreds of thousands of chest X-rays, not
on ImageNet, so what we would be buying is their pretraining, not their architecture. And CAM
adds up exactly to the prediction on this architecture, which a transformer does not give us.
A newer backbone is on the Phase 2 list, after the recipe.

**"Is 0.816 good?"**
It is the right number rather than a good one. DenseNet-121 in recent work runs 0.812 to 0.822,
so we are inside the range. The 224 run tells us where the remaining gap is: Goel gets 0.8202 at
224, where our recipe gets 0.8025, so what separates us from the best DenseNet-121 results is
the training recipe, not the network.

**"Why don't you report accuracy?"**
Because always answering "no finding" gets 0.92 on this dataset and our model gets 0.87. The
findings are rare, so accuracy rewards silence. We report AUROC, AUPRC, and precision and recall
at a cut-off chosen on validation.

**"Your localisation is below the published localisers."**
It is, and it should be. CLARiTy, ThoraX-PriorNet and the others are built to localise, with
anatomical priors, attention branches, or thresholds tuned on held-out boxes. Ours are unmodified
post-hoc maps from a plain classifier, which is what we set out to measure. We are level with
PCAN at the strictest threshold. CLARiTy also tunes on half the boxed images and scores the other
half; we score all 880 with no tuning.

**"What is novel here?"**
Not the classifier. The measurement: four methods and a random control, three tests including the
randomisation check most papers skip, on the official split, with the failures reported. Plus the
resolution result on nodules, which shows the standard box metric cannot judge the smallest
findings at map resolution.

**"How do we know the results are real?"**
Four runs are saved with weights, per-epoch history and test scores. 44 tests pass. Every table
in the report is regenerated by a script from those runs, and the commands are in the README.

**"What would change your mind about the approach?"**
If the built-in explanations in Phase 2 beat the post-hoc ones on all three tests, the honest
conclusion is that post-hoc maps should not be trusted on their own, and we would say so.

## The sixty second version, for a corridor

We trained a DenseNet-121 on NIH ChestX-ray14 at 512 pixels, on the official split, and we got
0.817 mean AUROC across three runs, which is where recent DenseNet-121 papers sit. The real work
is the audit. Four explanation methods plus a random control, each one scored three ways: does it
point where the radiologist drew the box, is it faithful to the model, and does it change when
the model is scrambled. The best method beats a random map by 2.4 times on pointing, no method
wins all three tests, and we report the failures. The scores are calibrated with a decision
cut-off per finding. Nothing touched the test set except the final evaluation.
