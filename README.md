# Leaky Distilled Monte Carlo (LDMC)

Code for the paper "Leaky Distilled Monte Carlo" (under review).

LDMC is a teacher–student framework for uncertainty estimation. An MC-based teacher
(MC Dropout or MC Laplace approximation) is quantized into a small set of control points,
and the student receives these control points through soft dropout layers.
This repository contains the two instantiations, QMCD (quantized MC Dropout) and
QMCLA (quantized MC Laplace approximation), and the DDU baseline.

## Structure

```
mnist/      qmcd/  qmcla/  ddu/
cifar10/    qmcd/  qmcla/  ddu/
svhn/       qmcd/  qmcla/
dbpedia/    qmcd/
```

Each experiment folder is self-contained. Run the scripts from inside the folder, since the
modules are imported by name.

| File | Content |
|---|---|
| `teacher.py` | Teacher network (ConvNet for MNIST, VGG-16 for CIFAR-10 and SVHN, DistilBERT for DBpedia-14) with the quantized dropout layers |
| `teacher_trainer.py` | Teacher training |
| `teacher_laplace.py` | KFAC Laplace approximation with eigenvector masks (QMCLA) |
| `student.py` | Student network with soft dropout layers that receive the control points |
| `evaluation.py` | KL divergence, RMSE, accuracy and AUROC per known/unknown class |
| `main.py` | Teacher training (if no checkpoint exists), student distillation, and evaluation |
| `ddu/main_ddu.py` | DDU baseline |

SVHN uses separate scripts for the teacher-side evaluation and the student:

```
cd svhn/qmcd
python teacher_chart_qmcd.py <p_drop> <p_rw>          # train and evaluate the QMCD teacher
python main_student.py <p_drop> <p_rw> <True|False|rand>

cd svhn/qmcla
python teacher_chart.py                                # train and evaluate the QMCLA teacher
python main_student.py <p_m> <True|False|rand>
```

The last argument selects the student: `True` is LDMC, `False` is standard single-pass
distillation, and `rand` is the ablation with randomly selected control points.

## Setup

```
pip install -r requirements.txt
```

MNIST, CIFAR-10 and SVHN are downloaded by torchvision into `./data`,
DBpedia-14 is downloaded with Hugging Face `datasets`, and Ambiguous-MNIST with `ddu_dirty_mnist`.

Before the first run, create the checkpoint folder inside the experiment folder:

```
mkdir -p model
python main.py
```
