// Config
#set document(title: [Rapport de mi-projet], author: "Anthony Fernandes")
#set text(lang: "fr")
#set par(justify: true)

#show title: set text(fill: rgb("#8A1538"))
#show heading: set text(fill: rgb("#8A1538"))
#show link: set text(fill: blue)
#show cite: set text(fill: blue)
#show raw.where(block: false): it => box(
  it, fill: luma(90%),
  inset: (x: 2pt),
  outset: (y: 2pt),
  radius: 25%
)

// Report

#align(center)[
  #title()

  _Anthony Fernandes, 22201572, MIDS_
]

Le code de ce projet est disponible sur ce dépôt github : https://github.com/yapudpill/optim-projet.git. Le code des tests de cette première partie sont dans le répertoire `tests`, le code du rendu final se trouvera dans le répertoire `src`.

= Travail effectué à mi-parcours

Cette première moitié de projet a surtout été une phase d'expérimentation pour prendre en main pytorch, lire sa documentation et comprendre les différentes notions vues en cours.

J'ai commencé par concevoir différentes situations d'optimisation et de machine learning afin de comparer les comportements de la descente de gradient ordinaire et de la descente à pas spectral. La plupart de ces tests n'ont pas réussis à mettre en avant un avantage significatif de specGD, voire ont montrer qu'elle est moins performante que la descente classique. Voici un florilège de ces tests :

#grid(columns: 2, gutter: 1em)[
  #figure(
    image("quadratic.png"),
    caption: [
      On voit $X in cal(M)_n (RR)$ comme un élément de $RR^n^2$. On minimise la fonction quadratique $F(X) = 1/2 chevron.l A X, X chevron.r$ avec $A = "diag"(1, 100, 1, ..., 1)$ de conditionnement $100$.
    ]
  )
][
  #figure(
    image("regression.png"),
    caption: [
      On considère un modèle gaussien $Y = X theta^top + epsilon$ avec $N$ variables explicatives et $M$ variables à prédire. On calcule l'estimateur des moindres carrés $"argmin"_theta 1/2 norm(Y - X theta^top)^2$.
    ]
  )
][
  #figure(
    image("cos_nn.png"),
    caption: [
      On apprend un réseau de neurones à une couches cachée la fonction $f(x, y) = cos(x + y)$.
    ]
  )
][
  #figure(
    image("exemple_cours.png"),
    caption: [
      On minimise $F(U) = 1/4 norm(U U^top - M^star)^2_F$ avec $M^star$ symétrique définie positive.
    ]
  )
]

Suivant ces tests non concluants, j'ai cherché de la littérature sur le sujet. J'ai trouvé un papier @davis2026 selon lequel le pas spectral est plus performant pour l'optimisation de matrices suivant une fonction d'activation, i.e. les poids cachés d'un réseau de neurones. Pour tester cet aspect j'ai implémenté un réseau de neurones à trois couches cachées et lui ai fait apprendre à dessiner 10 points équidistants sur des cercles de rayons et angles différents. L'entrainement a été fait via GD sur tous les poids sauf un, qui a été optimisé avec specGD. En variant le poids optimisé avec specGD, on constate effectivement que la descente spectrale est plus performante que la descente classique quand elle est appliquée sur les poids vers la fin du réseau.

#figure(
  image("circles.png"),
  caption: [
    On apprend un réseau de neurones à quatre couches cachées à dessiner 10 points équidistants sur des cercles de rayons et angles différents. Chaque sous-figure correspond à l'entrainement du réseau avec un poids différent optimisé par specGD.
  ]
)

= Difficultés et incertitudes rencontrées

N'ayant jamais fait de machine learning avant ce projet, j'ai eu du mal à concevoir des situations interessantes pour tester la descente spectrale et ne suis encore pas sûr qu'elles le sont.

Aussi, ne sachant pas vraiment où j'allais, j'ai préféré ne pas considérer les extensions de la descente comme l'ajout d'inertie ou de stochasticité. Ces variations ajoutent de nouveaux hyperparamètres à ajuster, ce qui aurait rendu les tests encore plus difficiles à interpréter.

= Objectifs pour la suite du projet

Les tests non concluants montrent tout de même un phénomène interessant : à chaque fois on obtient une décroissance raide de la fonction à optimiser suivi d'un plateau. En augmentant la valeur du pas on peut rendre la décroissance plus raide mais le plateau apparait à une valeur plus grande. Ceci est probablement dû au fait que la taille de la mise à jour n'est pas décroissante avec le nombre d'itérations.

$
  norm(eta "polar"(nabla F(W_k)))_F = eta sqrt("rg"(nabla F(W_k)))
$

Le rang du gradient est une quantité qui n'a pas de raison de décroître avec $k$. Pour parer à ce problème, on peut utiliser la mise à jour suivante @davis2026 :

$
  W_(k+1) = W_k - eta norm(nabla F(W_k))_* "polar"(nabla F(W_k))
$

J'aimerai essayer de programmer cette mise à jour et voir si elle permet d'obtenir de meilleures performances sur mes tests précédents ou sur de nouveaux tests.

= Remarques

- Ce rapport fait bien une page sans considérer les figures.
- Aucune IA générative n'a été utilisé pour la rédaction de ce rapport et le code de ce projet, sauf pour la génération des figures.

#bibliography("bib.bib")
