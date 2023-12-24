# Devil Palette
## La palette de couleurs analytique

Ce petit module permet d'obtenir un rapide résumé des couleurs principales d'une image comme suit :

<img src="https://github.com/raphadasilva/chambre_noire-/blob/main/devil_palette/img/palette.jpg?raw=true" alt="palette_simple" width="300"/>

Mais ce module peut aussi produire une *palette à la diable* plus analytique telle que :

<img src="https://github.com/raphadasilva/chambre_noire-/blob/main/devil_palette/img/devil_palette.jpg?raw=true" alt="devil_simple" width="300"/>

Le plus simple pour l'utiliser dans un Colab revient tout simplement à cloner ce repo via la commande :

```sh
!git clone https://github.com/raphadasilva/devil_palette.git
```

Puis de l'importer :

```sh
import devil_palette.devil_palette as dp
```
Deux fonctions principales sont fournies :
* simple_palette(url)
* devil_palette(url)

A noter que chaque génération de palettes est associée avec un fichier txt listant les couleurs impliquées.

Pour plus de détails sur la méthodologie, [ce calepin est à disposition]([https://github.com/raphadasilva/chambre_noire-/blob/main/devil_palette/img/palette.jpg](https://github.com/raphadasilva/chambre_noire-/blob/main/devil_palette/devil_palette_analyse_et_discretisation_des_couleurs_d'une_image_PIL_analysis_picture_colors.ipynb)https://github.com/raphadasilva/chambre_noire-/blob/main/devil_palette/devil_palette_analyse_et_discretisation_des_couleurs_d'une_image_PIL_analysis_picture_colors.ipynb).
