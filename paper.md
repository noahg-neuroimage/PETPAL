# PETPAL: Positron Emission Tomography Processing Analysis Library

## Summary

A toolbox for processing PET (Positron Emission Tomography) data. A non-exhaustive list of capabilities:

- Decay correction
- Motion correction
- Anatomical co-registration
- Image summing
- SUV
- SUVR
- Calculate TACs
- Kinetic modeling by ROI
- Kinetic modeling by voxel, parametric imaging
- Regional PET stats
- AIF preprocessing
- IDIF estimation
- Partial Volume Correction (PVC) using Symmetric Geometric Transfer Matrix (sGTM)

PETPAL is intended for research applications, such as kinetic modeling of new radiotracers,
determination of reference region, evaluation of uptake differences in a disease population.

## Statement of Need

PETPAL provides researches with accessible tools for creating a pipeline for processing PET data.

## State of the Field

While other PET processing software exist, there is a need for a complete, open source PET library
primarily focused on research applications. Some existing PET softare are primarily concerned with
clinical applications, require licensing fees, or are limited to specific steps in the PET processing
pipeline. PETPAL fills a gap in this software space. It is free and open source, covers a wide variety
of PET related processing tasks, and is designed by PET researchers for the purpose of doing PET
research.

## Software Design

PETPAL is written in Python 3.12. One of the most widely used programming languages, Python allows
for easy integration with other research software, ease of use, and facilitation of user
contributions.

The API is based on command line interface (CLI) wrappers of Python code. The interface is designed
to provide users with as many options as possible, rather than making assumptions on the user's
behalf. For example, the user can use their preferred method of segmenting anatomical imaging
when applying it to kinetic modeling. This also allows for users to set up their own pipeline that
integrates other software as needed for their specific research needs.

## Research Impact Statement

PETPAL was used to process PET data in a study of a new radiotracer, VAT, in dystonia, a disease
population. This research is in preparation for publication. PETPAL has been used in processing of
preliminary data for NIH grant appliations (such as ). Research results obtained with PETPAL have
been presented at the Movement Disorders Society (MDS) yearly meeting (citation).

## AI use disclosure

Generative AI was used sparingly in the creation of the PETPAL software. It was used to generate
some unit tests for PETPAL functions. 

