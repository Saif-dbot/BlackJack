Installation
============

Prérequis
---------

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- (Optionnel) CUDA pour l'entraînement accéléré des DQN

Installation via pip
--------------------

1. Cloner le repository:

.. code-block:: bash

   git https://github.com/Saif-dbot/BlackJack
   cd PBlackJack

2. Créer un environnement virtuel:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate    # Windows

3. Installer les dépendances:

.. code-block:: bash

   pip install -r requirements.txt

4. Installer le package en mode développement:

.. code-block:: bash

   pip install -e .

Dépendances Principales
-----------------------

- **NumPy** (≥1.21.0): Calculs numériques
- **PyTorch** (≥1.10.0): Réseaux de neurones pour DQN
- **Streamlit** (≥1.20.0): Interface web interactive
- **Matplotlib** (≥3.5.0): Visualisations
- **Plotly** (≥5.0.0): Graphiques interactifs
- **Pandas** (≥1.4.0): Manipulation de données
- **PyYAML** (≥6.0): Configuration

Vérification de l'Installation
-------------------------------

Pour vérifier que l'installation est réussie:

.. code-block:: python

   python -c "import src; print('Installation réussie!')"

Configuration Optionnelle
-------------------------

CUDA
~~~~

Pour utiliser CUDA avec PyTorch:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Vérifier CUDA:

.. code-block:: python

   import torch
   print(f"CUDA disponible: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")

Problèmes Courants
------------------

**Import Error sur streamlit**
   Réinstaller streamlit: ``pip install --upgrade streamlit``

**Erreur PyTorch CUDA**
   Installer la version CPU: ``pip install torch --index-url https://download.pytorch.org/whl/cpu``

**Module 'src' non trouvé**
   Installer en mode développement: ``pip install -e .``
