# Compression-des-images-fixes-par-JPEG

On distingue les techniques de compression conservatrice qui permettent de
reconstituer, en fin de processus, une image identique à l’image initiale, et les
techniques de compression non conservatrice ou dites avec perte. Ce sont ces dernières
qui nous intéressent dans le cadre de ce TP. Ces méthodes peuvent réaliser une
compression très poussée, moyennant une certaine perte d’information. L’image
reconstituée en fin de processus diffère de l’image initiale, mais la différence de qualité
n’est pratiquement pas perçue par l’œil humain.
Le principe de la compression JPEG consiste à supprimer les détails les plus fins d'une
image, autrement dit, les détails que le système visuel humain ne peut détecter.
Nous allons dans ce TP réaliser la suite des opérations à effectuer pour compresser une
image en passant par les trois phases majeures de la compression d’images avec perte :

a. Transformée en cosinus discrète bi-dimensionnelle (DCT)
b. Quantification
c. Codage entropique
