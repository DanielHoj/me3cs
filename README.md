
<h2> 
<img src="https://github.com/DanielHoj/me3cs/blob/master/me3cs_logo.png" width="200"  align = "center">
 me3cs - the chemometrical package for python
</h2>



# Table of Contents
1. [What is it?](#What_is_it)
2. [Why use it?](#why_use_it)
3. [What can it do?](#what_can_it_do)
5. [But can you give an example?](#Example)
6. [Installation](#Installation)
7. [License](#License)

## What is it? <a name="What_is_it"></a>
*me3cs* is a high level, centralised framework for chemometrical data analysis in python. It is designed to be simple on the surface, while including the functionality you would need.

## Why use it? <a name="why_use_it"></a>
*me3cs* allows you to spend more time looking at data and the results, and less time coding (even though coding is fun, it can be time consuming..) 
This enables you to use an iterative analysis workflow, and gives you an quick and easy way to compare model diagnostics and results.
The following image illustrates a typical workflow with *me3cs*:\
<img src="https://github.com/DanielHoj/me3cs/blob/master/flowcharts/me3cs_workflow.jpg" width="850">

So do you want the endless possibilities python offers, but miss a quicker way of doing chemometrical data analysis: *me3cs* is what you need.

## What can it do? <a name="what_can_it_do"></a>
Main features:
* A high level, centralised workframe, enabling a fast and iterative analysis workflow. 
* A preprocessing module, including scaling, filtering, normalisation and standarisation methods.
* A missing values module with different imputation and interpolation algorithms and a quick way of removing `NaN` values.
* Machine learning models for different use-cases; regression, classification or exploratory analysis.
* Built-in cross-validation methods, such as venetian blinds or contiguous blocks.
* Outlier detection methods and quick removal of observational or variabel outliers.
* A logging system, that keeps track of the models you have created.
* All algorithms are written with numpy or scipy, avoiding major dependencies.
* Supports data as pandas DataFrames and Series, or numpy ndarrays.

The structure of a *me3cs* model is illustrated in the following image:\
<img src="https://github.com/DanielHoj/me3cs/blob/master/flowcharts/me3cs - Model Framework.jpg" width="500">

## But can you give an example?? <a name="Example"></a>
Yes, you bet!

The following example uses the GluFrucSuc data set from the University of Copenhagen: 

https://food.ku.dk/english/research_at_food/research_fields/foodomics/algorithmsandsoftware/ 

The data is NIR measurements of mixture samples with glucose, fructose and succrose. The example illustrates a quick way to make a pls-r model. If other objectives are desired, the appropriate models can likewise be created.


***Loading the data:***

The matlab data is loaded as numpy objects, with x as the nir spectra and y as the fructose reference values:

    from scipy.io import loadmat
    import numpy as np
    data = loadmat(
        "/dataset_2_glufrusuc.mat"
    )
    x = data["data"]
    y = data["fructose_ref"]

An instance of a *me3cs* model is created:

     import me3cs as m3
     mdl = m3.Model(x,y)
     
 ***Creating your first model:***

A PLS model can be created simply by calling the pls method:

     mdl.pls()
     
The cross-validation default settings are *venetian blinds*, but can be changed in ``mdl.options.cross_validation``. If the data has not been preprocessed by a *scaling* method, the data is mean centered prior to creating a PLS model.



After the PLS model is created, the results can be found in ``mdl.results``. The optimal number of components can be decided by looking at the *rmse-cv* and *rmse-c* values. These can be found under ``mdl.results.calibrations.rmse`` and ``mdl.results.cross_validation.rmse``, and can visually be compared with a third-party plotting library such as ``matplotlib``. The optimal number of components can be set in ``mdl.results.optimal_number_component``, but an initial guess is provided when the PLS model is made.  $Q-residuals$ and $hotelling\ T\^2$ values can be found in ``mdl.results.diagnostics``. To remove possible outliers the module ``mdl.results.outlier_detection`` can be used. 

    mdl.results.outlier_detection.remove_outlier_from_hotellings_t2()

removes the observation with the highest $hotelling\ T\^2$ value at the optimal number of component. Observations can also be removed based on $Q-residuals$ and leverage values. A new model can be calculated from the subset of data, once again calling the 

    mdl.pls()
    
The number of observations can be reset with the ``mdl.results.outlier_detection.reset()``method.

The x and y data can be preprocessed by using the ``preprocessing`` module. For instance

    mdl.x.preprocessing.msc()
    
performs *multiplicative scatter correction* on the x data. The preprocessing is sorted so that *scaling* methods are called lastly. A list of the called preprocessing methods can be found under ``mdl.x.preprocessing.called.function``. To reset the called preprocessing methods ``mdl.x.preprocessing.reset()`` can be called.

     mdl.pls()
     
A new pls model has now been made with the x data being preprocessed. The model can once again be investigated regarding outliers.

If the data contain ``NaN`` values, the ``missing_data`` module can be used to either remove the values, or create new values based on a set of algorithms.

The regression coefficients of the calibrated the model can be found under ``mdl.results.calibrations.reg``, from which new predictions can be made.
   

## Installation <a name="Installation"></a>

## License <a name="License"></a>
