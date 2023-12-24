import PIL, re, requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def col_to_text(DF:pd.core.frame.DataFrame,n_col="hex",path_file="palette.txt"):
    """
        Cette fonction transforme la colonne d'une DataFrame en chaîne de caractères glissée dans un fichier txt.
    
    """
    try:
        with open(path_file,"w") as f:
            f.write(r' / '.join(DF[n_col]))
        f.close()
    except IndexError:
        print("Merci de considérer une DataFrame contenant la colonne mise en argument")


def open_pic(path:str):
    """
        Cette fonction renvoie une image, si la chaîne de cara en paramètre pointe vers un fichier jpeg ou png valide.
        nb : marche aussi avec les URL
    """
    url_reg = r"^http[s]{,1}.*\.(([jJ][pP][Ee]{,1}[Gg])|([Pp][Nn][Gg]))$"
    file_reg = r".*\.(([jJ][pP][Ee]{,1}[Gg])|([Pp][Nn][Gg]))$"
    try:
        if re.match(url_reg,path):
            img = requests.get(path)
            return Image.open(BytesIO(img.content))
        elif re.match(file_reg,path):
            return Image.open(path)
        else:
            print("Merci de renseigner un chemin/URL pointant vers un fichier jpg ou png !")
    except requests.ConnectionError:
        print("Attention, l'URL renseignée ne semble pas pointer vers un serveur.")
    except PIL.UnidentifiedImageError:
        print("Attention, le chemin renseigné ne semble pas pointer vers un jpeg ou png valide !")


def rgb_convert(path:str):
    """
        Cette fonction convertit une image au format RGB.
    """
    try:
        return open_pic(path).convert('RGB')
    except:
        print("Quelque chose a coincé, peut-être à cause d'une corruption du fichier image.")


def pic_newd(img:Image,width=512):
    """
        Cette fonction renvoie la version réduite d'une image en conservant le ratio largeur/hauteur.
        IMPORTANT : le premier paramètre doit bien correspondre à la nouvelle largeur en pixels, pas à la hauteur.
    """
    try:
        ratio = img.size[0]/img.size[1]
        return img.copy().resize((width,int(round(width/ratio))))
    except TypeError:
        print("""Attention aux formats de vos variables. Pour rappel :
             width doit être un nombre
             img doit correspondre à une image digérée par le module PIL""")

def pic_matrix(img:Image):
    """
        Cette fonction renvoie une matrice Numpy à partir d'une image RGB taillée pour être transformée en tableaux Numpy.
    """
    m_array = []
    try:
        m_array = np.asarray(img).copy()
    except PIL.UnidentifiedImageError:
        print("Attention, le chemin renseigné ne semble pas pointer vers un jpeg ou png valide !")
    return m_array

def hex_convert(arr:np.array, state="picture"):
    """
        Cette fonction convertit un array Numpy rempli de triplets R,G,B en codes hexadécimaux.
    """
    try:
        match(state):
            case "picture":
                transpo = np.dstack((arr[:,:,0].flatten(),arr[:,:,1].flatten(),arr[:,:,2].flatten()))[0,:,:]
                return [f"#{bytes(t).hex().upper()}" for t in transpo]
            case "clusters":
                return ['#%02X%02X%02X' % tuple(t) for t in arr]
            case _:
                print("Attention, seul deux cas sont paramétrés pour state : 'picture' pour un array non rangé")
    except TypeError:
        print("Attention au format de la variable : il faut un tableau Numpy content des triplets RGB valides tels que [0,255,0]")
    except IndexError:
        print("Attention à la taille de vos variables")

def prep_picture(path:str, width=512):
    """
        Cette fonction prépare une image en vue d'un retraitement par intelligence artificielle.
        Elle ne prend que deux arguments : un chemin (local ou URL) et une largeur
    
    """
    try:
        img = pic_newd(rgb_convert(path),width)
        if max(img.size)<256:
            print("Attention, l'image est peut-être de trop basse résolution pour donner des palettes pertinents")
        elif min(img.size)>1024:
            print("L'image est d'assez haute résolution. Nous conseillons de la réduire afin d'obtenir une palette plus pertinente.")
        DF = col_neslist(pd.DataFrame(columns=["r","v","b","hex"]),pic_matrix(img),["r","v","b"])
        DF["hex"] = hex_convert(pic_matrix(img))
        return DF, img
    except TypeError:
        print("Veuillez vérifier le format de vos variables")

def round_list(t:list):
    """
        Cette fonction retourne une liste de nombres entiers arrondis à partir d'une liste de nombres à décimales.
    """
    try:
        return [int(round(e)) for e in t]
    except TypeError:
        print("Attention : votre liste de base ne doit contenir que des nombres !")


def empty_df(l_col:list):
    """
        Cette fonction renvoie une DataFrame vide dont les colonnes sont dictées par une liste choisie par l'utilisateur.
    """
    try:
        return pd.DataFrame(columns=l_col)
    except TypeError:
        print("Merci de bien mettre une liste en unique argument de cette fonction")

def col_neslist(DF:pd.core.frame.DataFrame,neslist:np.array,col:list):
    """
        Cette fonction crée ou met à jour les colonnes d'une DF en fonction d'arrays Numpy imbriqués interrogés par position.
        Ces derniers doivent tous avoir la même longueur, et cette longueur doit être égale à celle de la list col.
    """
    try:
        if len(neslist.shape) == 3:
            for i,k in enumerate(col):
                DF[k] = neslist[:,:,i].flatten()
        else:
            for i,k in enumerate(col):
                DF[k] = neslist[:,i]
        return DF
    except TypeError:
        print("Attention aux format des variables : DF = DataFrame / nestlist = arrays Numpy imbriqués / col = liste")
    except IndexError:
        print("Attention à la taille de la liste col : elle doit être de la même longueur que les tableaux imbriqués !")
    except ValueError:
        print("Attention aux tailles comparées : la DF a trop de lignes en comparaison des colonnes que vous essayez de mettre à jour")


def unique_DF(DF:pd.core.frame.DataFrame,target:list,u_type="kfirst"):
    """
        Cette fonction renvoie une copie d'une DataFrame expurgés des doublons sur une colonne cible
    """
    UDF = DF.copy()
    try:
        match(u_type):
            case "kfirst":
                UDF = UDF.drop_duplicates(subset=target).reset_index(drop=True)
            case "sunique":
                UDF = UDF.drop_duplicates(subset=target,keep=False,ignore_index=True)
            case _:
                print("""Attention, seuls deux cas sont paramétrés pour u_type : 
                    u_type='kfirst' pour garder toutes les valeurs apparaissant au moins une fois
                    u_type = 'sunique' pour ne garder que les valeurs qui n'apparaissent qu'une fois
                """)
    except TypeError:
        print("Merci de bien respecter les formats de variables attendues par la fonction")
    except KeyError:
        print("Merci de bien renseigner une liste de colonnes toutes présentes dans la DataFrame à écrémer !")
    return UDF


def nclusters_kmeans(X:pd.core.frame.DataFrame):
    """
        Cette fonction renvoie un nombre "optimal" de clusters pour un algo K-Means. Ce nombre est déterminé en utilisant un "coude"
        retenu par le module yellowbrick.
    """
    print("Ca va prendre moins d'une minute, mais buvez du café !")
    try:
        draft = KElbowVisualizer(KMeans(n_init=10), k=(2,11))
        draft.fit(X)
        return draft.elbow_value_
    except:
        print("Attention au format attendu pour X : il s'agit bien d'une DF avec des Series chiffrées uniquement")


def clusters_to_df(modele:KMeans):
    """
        Cette fonction transforme un modèle de clustering en DF contenant 
         - des colonnes R, V, B, 
         - une colonne des codes hexadécimaux correspondants
         - le code de cluster      
    """
    try:
        centers = np.array([round_list(l) for l in modele.cluster_centers_])
        DF = empty_df(["r","v","b","hex"])
        DF["hex"] = hex_convert(centers,state="clusters")
        DF["cluster"] = DF.index
        DF = col_neslist(DF,centers,["r","v","b"])
        return DF
    except AttributeError:
        print("Attention à bien mettre en variable un modèle de clustering type K-Means")


def labels_kmeans(DF:pd.core.frame.DataFrame,modele:KMeans):
    """
      Cette fonction labellise une DF qui a servi d'apprentissage à un modèle KMeans par le biais d'une nouvelle colonne.  
    """
    try:
        DF["cluster"] = modele.labels_
        return DF
    except IndexError:
        print("Attention à la DataFrame d'apprentissage, elle n'est pas de la taille attendue par le modèle.")


def hexcent_clus(DF_dest:pd.core.frame.DataFrame,DF_source:pd.core.frame.DataFrame):
    """
      Cette fonction transforme une DataFrame contenant des clusters liés à des couleurs hexadécimales en pourcentages à partir de la DF d'apprentissage
    """
    try:
        DF_hexcent = DF_dest.copy()
        DF_hexcent = DF_hexcent[["hex","cluster"]]
        DF_hexcent["p"] = [round((len(DF_source[DF_source["cluster"]==c])/len(DF_source))*100,2) for c in DF_hexcent["cluster"]]
        DF_hexcent.sort_values(by=["p"], ascending=False, inplace=True)
        return DF_hexcent
    except IndexError:
        print("Attention, votre DF de clusters doit bien contenir deux colonnes hex et clusters.")


def prep_cluster(DF:pd.core.frame.DataFrame, n_clus=5, deter=False):
    """
        Cette fonction prépare trois DataFrames à partir d'un tableau Numpy généré par la fonction prep_picture.
        Elle va, en fonction d'un nombre de classes (entre 2 et 10), déterminer les groupes de couleurs les plus cohérents.
        On peut aussi laisser le programme déterminer le nombre de clusters optimal en changeant la variable deter. 
    """
    try:
        X = DF[["r","v","b"]]
        if deter:
            n_clus = nclusters_kmeans(X)
        elif ~n_clus in list(range(2,11)):
            n_clus = 5
            print("Vous devez choisir pour n_clus un nombre compris entre 2 et 10. Le nombre imposé sera 5.")
        modele = KMeans(n_clusters=n_clus,n_init=10).fit(X)
        DF_clusters = clusters_to_df(modele)
        DF = labels_kmeans(DF,modele)
        DF_hexcent = hexcent_clus(DF_clusters,DF)
        return DF, DF_clusters ,DF_hexcent
    except TypeError:
        print("Veuillez vérifier le format de vos variables")


def bar_pct(axis_ref,DF:pd.core.frame.DataFrame,orientation="horizontal"):
    """
        Cette fonction affiche une palette en diagramme proportionnel trié dans l'ordre décroissant.
        NB : il est conseillé de l'utiliser après un appel de hexcent_clus()
    """
    try:
        match(orientation):
            case "horizontal":
                axis_ref.set_xlim(0,100)
                plt.barh(0,DF.iloc[0]["p"],color=DF.iloc[0]["hex"])
                counter = DF.iloc[0]["p"].copy()
                for c,p in zip(DF.iloc[1:]["hex"],DF.iloc[1:]["p"]):
                    plt.barh(0,p,color=c,left=counter)
                    counter += p
                axis_ref.grid(False)
                axis_ref.axis('off')
            case "vertical":
                axis_ref.set_xlim(0,100)
                plt.bar(0,DF.iloc[0]["p"],color=DF.iloc[0]["hex"])
                counter = DF.iloc[0]["p"].copy()
                for c,p in zip(DF.iloc[1:]["hex"],DF.iloc[1:]["p"]):
                    plt.barh(0,p,color=c,bottom=counter)
                    counter += p
                axis_ref.grid(False)
                axis_ref.axis('off')
            case _:
                print("Attention, seul deux cas sont codés pour l'attribut orientation. Veuillez choisir entre horizontal et vertical")
    except IndexError:
        print("Soyez vigilants sur la DataFrame : elle doit bien contenir des colonnes 'hex' et 'p'")


def one_hexcent(axis_ref,DF:pd.core.frame.DataFrame,index_cluster:int, orientation="vertical"):
    """
        Cette fonction affiche (à l'horizontale ou à la verticale) le pourcentage d'un cluster précis, coloré et centré.
      Elle prend en argument un axe de référence sur une visualisation matplotlib, une DataFrame
    """
    try:
        match(orientation):
            case "vertical":
                row_clus = DF[DF["cluster"]==index_cluster]
                b_margin = abs((row_clus["p"]-100)/2)
                axis_ref.set_ylim(0,100)
                plt.bar(0,row_clus["p"],bottom=b_margin,color=row_clus["hex"])
                axis_ref.grid(False)
                axis_ref.axis('off')
            case "horizontal":
                row_clus = DF[DF["cluster"]==index_cluster]
                b_margin = abs((row_clus["p"]-100)/2)
                axis_ref.set_ylim(0,100)
                plt.barh(0,row_clus["p"],left=b_margin,color=row_clus["hex"])
                axis_ref.grid(False)
                axis_ref.axis('off')
            case _:
                print("L'attribut orientation ne prévoit que deux cas de figure. Merci de choisir entre horizontal et vertical.")
    except TypeError:
        print("Attention au format des variables.")
    except IndexError:
        print("Attention : votre DataFrame doit bien contenir une colonne 'cluster'")


def D3_clusters(axis_ref,colors:pd.core.frame.DataFrame,clusters:pd.core.frame.DataFrame,r_data=3,r_cluster=150):
  """
      Cette fonction affiche un nuage de points 3D à partir d'une DF de couleurs d'une image
      et des centres de clusters associés (eux aussi transformés en DF)
  """
  try:
    axis_ref.set_aspect('equal')
    axis_ref.scatter(colors["r"], colors["v"], colors["b"], marker="o", c=colors["hex"], s=r_data, alpha=.4, edgecolor="white",linewidth=.002, zorder=1)
    axis_ref.scatter(clusters["r"], clusters["v"], clusters["b"], marker="o", c=clusters["hex"], s=r_cluster, alpha=.9, edgecolor="black",linewidth=.5, zorder=2)
    axis_ref.grid(False)
    axis_ref.xaxis.pane.fill = False
    axis_ref.xaxis.pane.set_edgecolor('w')
    axis_ref.set_xlabel('R')
    axis_ref.yaxis.pane.fill = False
    axis_ref.yaxis.pane.set_edgecolor('w')
    axis_ref.set_ylabel('V')
    axis_ref.zaxis.pane.fill = False
    axis_ref.zaxis.pane.set_edgecolor('w')
    axis_ref.set_zlabel('B')
  except:
    print("Attention au format de vos variables !")


def clus_opa(img:Image, DF:pd.core.frame.DataFrame, index_cluster:int):
    """
        Cette fonction compare une image avec une DF de ses pixels divisés en clusters.
        Pour un cluster donné, la fonction opacifie tous les pixels qui ne font pas partie de l'image
    """
    try:
        t_img = img.copy().convert("RGBA")
        arr_img = np.array(t_img, dtype=np.ubyte)
        mask = ~np.array(DF["cluster"]==index_cluster).reshape(arr_img[:,:,3].shape)
        arr_img[:,:,-1] = np.where(mask, 0, 255)
        return Image.fromarray(np.ubyte(arr_img))
    except IndexError:
        print("Attention, votre DF doit avoir le même nombre de lignes que les pixels de l'image !")


def simple_palette(img_path:str,w=512,nb_clus=5, auto_cluster=False,path_palette="palette.jpg", path_text="palette.txt"):
    """
       Cette fonction crée une palette simple à partir d'une image locale ou hébergée sur Internet.
       Le nombre de couleurs de la palette (entre 2 et 10) est renseigné par l'utilisateur ou calculé par K-Means. 
       Il suffit de changer auto_cluster par True.
       Chaque palette est accompagnée du détail de ses couleurs dans un fichier texte.
    
    """
    try:
        DF, img_ref = prep_picture(img_path,width=w)
        hex_DF = prep_cluster(DF, n_clus=nb_clus,deter=auto_cluster)[2]
        
        fig = plt.figure(figsize=(15,15),constrained_layout=True)
        gspec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4,.3], hspace=.1)

        ax0 = fig.add_subplot(gspec[0])
        ax0.imshow(img_ref)
        ax0.axis('off')
        ax1 = fig.add_subplot(gspec[1])
        bar_pct(ax1,hex_DF)
    
        col_to_text(hex_DF,path_file=path_text)
        plt.savefig(path_palette,dpi=300, bbox_inches="tight", pad_inches=1)
    except TypeError:
        print("Attention à vos chemins de fichier.")


def devil_palette(img_path:str,w=512,nb_clus=5, auto_cluster=False,path_palette="devil_palette.jpg", path_text="palette.txt"):
    """
        Cette fonction crée une 'palette à la diable' à partir d'une image.
        Elle se compose d'un rappel de l'image et d'une ligne par classe de couleur identifiée.
        A chaque ligne, nous avons :
            le rappel de la couleur, avec longueur proportionnelle à sa présence dans l'image
            la composition du cluster en nuage de points 3D sur un axe RVB
            les pixels présents dans le cluster directement dans l'image
        Le programme peut aussi déterminer lui-même le nombre de clusters (entre 1 et 10) si passe la variable auto_cluster à True    
    """
    try:
        DF, img_ref = prep_picture(img_path,width=w)
        def_DF, clus_DF, hex_DF = prep_cluster(DF, n_clus=nb_clus,deter=auto_cluster)

        fig = plt.figure(figsize=(15,10+(len(clus_DF)*10)),constrained_layout=True)
        gspec = fig.add_gridspec(nrows=1+len(clus_DF), ncols=3, height_ratios=[8]+[5]*len(clus_DF),width_ratios=[.5,7,4])

        ax0 = fig.add_subplot(gspec[0,:])
        ax0.imshow(img_ref)
        ax0.axis('off')

        for i in range(len(clus_DF)):
            id_clus = hex_DF.iloc[i]["cluster"]

            ax_b = fig.add_subplot(gspec[i+1,0])
            one_hexcent(ax_b,hex_DF,id_clus)    

            ax_p = fig.add_subplot(gspec[i+1,1],projection='3d')
            D3_clusters(ax_p,def_DF[def_DF["cluster"]==id_clus], clus_DF[clus_DF["cluster"]==id_clus])
            ax_p.axis('off')

            ax_i = fig.add_subplot(gspec[i+1,2])
            ax_i.imshow(clus_opa(img_ref,def_DF,id_clus))
            ax_i.axis('off')
        
        col_to_text(hex_DF,path_file=path_text)
        plt.savefig(path_palette,dpi=300, bbox_inches="tight", pad_inches=1)
    except TypeError:
        print("Attention à vos chemins de fichier.")
