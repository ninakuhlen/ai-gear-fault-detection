# Projekt: Fehlererkennung in Getrieben

Mey et al. (2020) untersuchten die automatisierte Erkennung von Unwuchten in rotierenden Wellen mithilfe von Maschinellem Lernen und veröffentlichten hierfür einen spezifischen Datensatz. In diesem Kontext entwickeln und analysieren sie vier Klassifizierungsalgorithmen: Convolutional Neural Network (CNN), Fully-Connected Neural Network (FCNN), Hidden Markov Model und Random Forest. Diese Algorithmen unterscheiden zwischen den Zuständen "eine Unwucht liegt vor" und "keine Unwucht liegt vor". Das FCNN erreicht hier, als die erfolgreichste Methode, eine Vorhersagegenauigkeit von 98,5 %. <br>
Das vorliegende Projekt reproduziert zunächst das von Mey et al. beschriebene CNN. Auf Basis des FCNNs wird zudem ein Neuronales Netzwerk (NN) entwickelt, das eine Unterscheidung der im Datensatz angegebenen fünf Unwucht-Stufen leisten kann. Das CNN verarbeitet hierbei direkt Sensordaten. Für das FCNN überführt die Fast
Fourier Transform (FFT) die Daten vorab in den Frequenzbereich. <br>
Im Rahmen des Projektes sind so mehrere Python Module entstanden. Der vollständige Code ist in diesem GitHub Repository verfügbar.

## **Setup des Projekts**

### **1) Ein Virtual Environment anlegen**

Um ein Virtual Environment anzulegen, geben Sie `python -m <name> <path>` im Terminal ein. `<name>` ist der Name des Environments. Typisch sind die Bezeichnungen "venv" oder ".venv". `<path>` ist der Dateipfad, in dem das Environment angelegt werden soll. Die im Environment genutzte Python Version kann dem Befehl `python` angehangen werden, z. B. `python3.10`. Stellen Sie vorab sicher, dass die entsprechende Version auf Ihrem System installiert ist. \
Um in dem Environment arbeiten zu können, geben Sie im Terminal `<name>\Scripts\activate` ein. Dies aktiviert das Environment und im Terminal beginnt die Command Line mit `(<name>)`. Das Virtual Environment schließen Sie mit `deactivate` im Terminal. Unter Umständen müssen Sie in IPython Notebooks die Python Distribution des Virtual Environments auswählen. Die Schaltfläche *Change Kernel* finden Sie bei IPYNB Dateien in VS Code am oberen rechten Fensterrand. Selbiges gilt für PY Files: Hier wählen Sie die Schaltfläche *Select Interpreter* am unteren rechten Fensterrand.


### **2) Dependencies speichern und installieren**

Der Befehl `python -m pip freeze > requirements.txt` legt eine Liste aller im Virtual Environment installierten Python Packages an und speichert sie in der TXT File `requirements.txt`. Eine solche Liste erlaubt die Installation aller notwendiger Packages über den Befehl: `python -m pip install -r requirements.txt`.

### **3) IPYNB Dateien ausführen**

Die IPYNB Notebooks müssen im Falle des FFT-Nets in der Reihenfolge *fft_net_preprocessing.ipynb* $\rightarrow$ *fft_net.ipynb* ausgeführt werden. Bei der Ersstausführung von *fft_net_preprocessing.ipynb* muss der Boolean `DOWNLOAD` gesetzt sein.

## Quellen
### **Literatur**
Harris, C. R., Millman, K. J., Van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., Van Kerkwijk, M. H., Brett, M., Haldane, A., Del Río, J. F., Wiebe, M., Peterson, P., . . . Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Mey, O., Neudeck, W., Schneider, A. & Enge-Rosenblatt, O. (2020). Machine learning-based unbalance detection of a rotating shaft using vibration data. In 2020 25th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA) (S. 1610–1617). IEEE. https://doi.org/10.1109/etfa46521.2020.9212000

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal Of Machine Learning Research, 12, 2825–2830. https://doi.org/10.5555/1953048.2078195

### **Online**
Kuhlen, N., Beißner, L., Bergermann, T. & Wallrad, L. (2024). AI Gear Fault Detection (Version V1) [Computer Software]. https://github.com/ninakuhlen/ai-gear-fault-detection

Mey, O., Neudeck, W., Schneider, A. & Enge-Rosenblatt, O. (2022, 23. Februar). Vibration Analysis on Rotating Shaft. Kaggle. Abgerufen am 20. Oktober 2024, von https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft

NumPy Developers. (o. D.). Discrete Fourier Transform (Numpy.FFT) — NUMPY v2.2 Manual. https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft

Scikit-learn Developers. (o. D.). RobustScaler. Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html