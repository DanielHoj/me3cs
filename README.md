
<h2> 
<img src="https://github.com/DanielHoj/me3cs/blob/master/me3cs_logo.png" width="200"  align = "center">
 me3cs - the chemometrical package for python
</h2>



# Table of Contents
1. [What is it?](#What_is_it)
2. [Why use it?](#why_use_it)
3. [What can it do?](#What_can_it_do)
4. [Can you give an example?](#Example)
5. [Installation](#Installation)
6. [License](#License)

## What is it? <a name="What_is_it"></a>
*me3cs* is a high level, centralised framework for chemometrical data analysis in python. The main feature is a pipeline, which enhances the workflow of spectroscopical data analysis.


## Why use it? <a name="why_use_it"></a>
* A high level, centralised workframe, enabling a fast and iterative analysis workflow. 
* A preprocessing module, including scaling, filtering, normalisation and standarisation methods.
* A missing values module with different imputation and interpolation algorithms and a quick way of removing `NaN` values.
* Machine learning models for different use-cases; regression, classification or exploratory analysis.
* Built-in cross-validation methods, such as venetian blinds or contiguous blocks.
* Outlier detection methods and quick removal of observational or variabel outliers.
* All algorithms are written with numpy or scipy, avoiding major dependencies.
* Supports data as pandas DataFrames and Series, or numpy ndarrays.

## What can it do? <a name="What_can_it_do"></a>
For a quick illustration of the *metric* workframe architecture, look at the following flowchart: \
<img src="https://github.com/DanielHoj/me3cs/blob/master/flowcharts/model_framework.jpg" width="500">

## Can you give an example?? <a name="Example"></a>
Yes, you bet!

The following example uses the GluFrucSuc data set from the University of Copenhagen: 

https://food.ku.dk/english/research_at_food/research_fields/foodomics/algorithmsandsoftware/ 

The data is NIR measurements of mixture samples with glucose, fructose and succrose. The example illustrates a quick way to make a pls-r model.

The matlab data is loaded as numpy objects, with x as the nir spectra and y as the fructose reference values:

    from scipy.io import loadmat
    import numpy as np
    data = loadmat(
        "/dataset_2_glufrusuc.mat"
    )
    x = data["data"]
    y = data["fructose_ref"]

An instance of *me3cs* model is created:

     import me3cs as m3
     mdl = m3.Model(x,y)
    
A PLS model can be created with the pls method:

     mdl.pls()
     
The cross-validation default settings are *venetian blinds*, but can be changed in ``mdl.options.cross_validation``. If the data has not been preprocessed by a *scaling* method, the data is mean centered prior to creating a PLS model.

After the PLS model is created, the results can be found in ``mdl.results``. The optimal number of components can be decided by looking at the *rmse-cv* and *rmse-c* values. These can be found under ``mdl.results.calibrations.rmse`` and ``mdl.results.cross_validation.rmse``, and can visually be compared with a third-party plotting library such as ``matplotlib``. The optimal number of components can be set in ``mdl.results.optimal_number_component``, but an initial guess is provided when the PLS model is made.  $Q-residuals$ and $hotelling\ T\^2$ values can be found in ``mdl.results.diagnostics``. To remove possible outliers the module ``mdl.results.outlier_detection`` can be used. 

    mdl.results.outlier_detection.remove_outlier_from_hotellings_t2()

removes the observation with the highest $hotelling\ T\^2$ value at the optimal number of component. A new model can be calculated from the subset of data, with the ``pls()`` method again. Observations can also be removed based on residual and leverage values. The number of observations can be reset with the ``mdl.results.outlier_detection.reset()``method.

The x and y data can be preprocessed by using the ``preprocessing`` module. For instance

    mdl.x.preprocessing.msc()
    
performs *multiplicative scatter correction* on the x data. The preprocessing is sorted so that *scaling* methods are called lastly. A list of the called preprocessing methods can be found under ``mdl.x.preprocessing.called.function``.

To reset the called preprocessing methods ``mdl.x.preprocessing.reset()`` can be called.

     mdl.pls()
     
A new pls model has now been made with the x data being preprocessed. The model can once again be investigated regarding outliers.

If the data contain ``NaN`` values, the ``missing_data`` module can be used to either remove the values, or create new values based on a set of algorithms.

The regression coefficients of the calibrated the model can be found under ``mdl.results.calibrations.reg``.
   

## Installation <a name="Installation"></a>

## License <a name="License"></a>
