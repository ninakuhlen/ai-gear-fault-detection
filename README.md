<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bilder nebeneinander</title>
    <style>
        .image-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 100%;
        }
        .images {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }
        .images img {
            max-width: 100%;
            height: auto;
        }
        .caption {
            text-align: center;
            margin-top: 10px;
            font-style: italic;
        }
    </style>
</head>


# Projekt: Fehlererkennung in Getrieben

In diesem Projekt wird 


[Vibration Analysis on Rotating Shaft](https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft?resource=download#)
[Paper](https://arxiv.org/pdf/2005.12742)


<div class="image-container">
    <div class="images">
        <img src="data\images\Measurement_Setup.png" alt="Bild 1" width=30%>
        <img src="data\images\Measurement_Setup_Block_Diagram.png" alt="Bild 2" width=30%>
    </div>
    <div class="caption">Caption</div>
</div>
    
> 
> - The vibration is defined as cyclic or oscillating motion of a machine or machine component from its position of rest.
> 
> - The use of machinery vibration and the technological advances that have been developed over the years, that make it possible to not only detect when a machine is developing a problem, but to identify the specific nature of the problem for scheduled correction.
> 
> - Fault detection at rotating machinery with the help of vibration sensors offers the possibility to detect damage to machines at an early stage and to prevent production down-times by taking appropriate measures.
> 

## **Setup des Projekts**
### **Virtual Environments**

#### **Ein Virtual Environment anlegen**

Um ein Virtual Environment anzulegen, geben Sie <code>python -m \<name> \<directory></code> im Terminal ein. <code>\<name></code> ist der Name des Environments. Typisch sind die Bezeichnungen "venv" oder ".venv". <code>\<directory></code> ist der Pfad, in dem das Environment angelegt werden soll. Die im Environment genutzte Python Version kann dem Befehl <code>python</code> angehangen werden, z. B. <code>python3.10</code>. Stellen Sie vorab sicher, dass die entsprechende Version auf Ihrem System installiert ist. \
Um in dem Environment arbeiten zu können, geben Sie im Terminal <code>\<name>\\Scripts\\activate</code> ein. Dies aktiviert das Environment und im Terminal beginnt die Command Line mit <code>(\<name>)</code>. Das Virtual Environment schließen Sie mit <code>deactivate</code> im Terminal. Unter Umständen müssen Sie in IPython Notebooks die Python Distribution des Virtual Environments auswählen. Die Schaltfläche *Change Kernel* finden Sie bei IPYNB Dateien in VS Code am oberen rechten Fensterrand. Selbiges gilt für PY Files: Hier wählen Sie die Schaltfläche *Select Interpreter* am unteren rechten Fensterrand.


#### **Dependencies speichern und installieren**

Der Befehl <code>python -m pip freeze > requirements.txt</code> legt eine Liste aller im Virtual Environment installierten Python Packages an und speichert sie in der TXT File <code>requirements.txt</code>. Eine solche Liste erlaubt die Installation aller notwendiger Packages über den Befehl: <code>python -m pip install -r requirements.txt</code>.

## **Convolutional Neural Network**

<div class="image-container">
    <div class="images">
        <img src="data\images\CNN_Structure.png" width=30%>
    </div>
    <div class="caption">Skizze der Architektur des im Paper verwendeten Convolutional Neural Networks. N<sub>conv</sub> ist die Anzahl der Hidden Layer aus jeweils Convolution, Normalization, Activation und MaxPooling.</div>
</div>


## **FFT-Net**

<div class="image-container">
    <div class="images">
        <img src="data\images\FFT_Net_Structure.png" width=30%>
    </div>
    <div class="caption">Skizze der Architektur des im Paper verwendeten Neural Networks zur Klassifizierung der FFT-transformierten Samples. N<sub>hidden</sub> ist die Anzahl der Hidden Layer aus jeweils Fully Connected Layer und Activation.</div>
</div>

1. Berechnung der FFT für jedes Fenster von einer Sekunde Dauer bzw. 4096 Samples der Messreihe des ersten Vibrationssensors ("Vibration_1"). Das Ergebnis sind jeweils 2048 Fourier-Koeffizienten pro Fenster.
2. Aufteilen der Daten in 90 % Training Data und 10 % Test Data.
3. Skalierung der FFT Daten: 
Afterwards, the FFT data were scaled
as follows: For each Fourier coefficient, the respective median
and interquantile spacing of quantiles 5 and 95 was calculated
based on the extent of the training dataset (2048 values for
the median and the interquantile spacing, respectively). The
median values were then subtracted from the FFT values
and the result was divided by the interquantile values. Fully
connected (FC) neural networks were then trained on the
training data.
4. Training des Fully Connected Neural Networks (FCNN)
The input consisting of 2048 Fourier
coefficients in each sample was followed by Nhidden hidden
and fully connected layers with LeakyReLU activation and
the output layer. Neural networks of this type with zero
(equivalent to logistic regression) to four hidden layers were
trained using the respective training data.


#### **Quellen**

Mey, O., Neudeck, W., Schneider, A. & Enge-Rosenblatt, O. (2020). Machine learning-based unbalance detection of a rotating shaft using vibration data. In 2020 25th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA) (S. 1610–1617). IEEE. https://doi.org/10.1109/etfa46521.2020.9212000

