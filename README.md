# Projekt: Fehlererkennung in Getrieben

In diesem Projekt wird 


[Vibration Analysis on Rotating Shaft](https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft?resource=download#)
[Paper](https://arxiv.org/pdf/2005.12742)

> 
> - The vibration is defined as cyclic or oscillating motion of a machine or machine component from its position of rest.
> 
> - The use of machinery vibration and the technological advances that have been developed over the years, that make it possible to not only detect when a machine is developing a problem, but to identify the specific nature of the problem for scheduled correction.
> 
> - Fault detection at rotating machinery with the help of vibration sensors offers the possibility to detect damage to machines at an early stage and to prevent production down-times by taking appropriate measures.
> 

### **Virtual Environments**

#### **Ein Virtual Environment anlegen**

Um ein Virtual Environment anzulegen, geben Sie <code>python -m \<name> \<directory></code> im Terminal ein. <code>\<name></code> ist der Name des Environments. Typisch sind die Bezeichnungen "venv" oder ".venv". <code>\<directory></code> ist der Pfad, in dem das Environment angelegt werden soll. Die im Environment genutzte Python Version kann dem Befehl <code>python</code> angehangen werden, z. B. <code>python3.10</code>. Stellen Sie vorab sicher, dass die entsprechende Version auf Ihrem System installiert ist. \
Um in dem Environment arbeiten zu können, geben Sie im Terminal <code>\<name>\\Scripts\\activate</code> ein. Dies aktiviert das Environment und im Terminal beginnt die Command Line mit <code>(\<name>)</code>. Das Virtual Environment schließen Sie mit <code>deactivate</code> im Terminal. Unter Umständen müssen Sie in IPython Notebooks die Python Distribution des Virtual Environments auswählen. Die Schaltfläche *Change Kernel* finden Sie bei IPYNB Dateien in VS Code am oberen rechten Fensterrand. Selbiges gilt für PY Files: Hier wählen Sie die Schaltfläche *Select Interpreter* am unteren rechten Fensterrand.


#### **Dependencies speichern und installieren**

Der Befehl <code>python -m pip freeze > requirements.txt</code> legt eine Liste aller im Virtual Environment installierten Python Packages an und speichert sie in der TXT File <code>requirements.txt</code>. Eine solche Liste erlaubt die Installation aller notwendiger Packages über den Befehl: <code>python -m pip install -r requirements.txt</code>.

