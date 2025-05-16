L’objectif de cette étude était d’évaluer et comparer   trois approches probabilistic de quantification d'incertitude  a savoir :  le modèle Single Network, Monte Carlo Dropout et Deep Ensemble – sur trois axes principaux :

1. La performance de classification,
2. La qualité de la calibration et de l'incertitude,
3. L'intégration de méthodes de prédiction conforme adaptative, pour mieux exprimer le doute.

Nous avons mené une série d’expériences rigoureuses sur le dataset SIPaKMeD, et voici les résultats obtenus à chaque étape.




# 🎤 🔵 SLIDE : Calibration Performance (ECE – Brier Score – Reliability Diagram)


J'ai ici évalué la calibration des probabilités des modèles, à l’aide de l’ECE, du Brier Score, et des Reliability Diagrams.

`ECE` ≈ 0.02 pour tous → Les trois modèles sont bien calibrés en apparence.

`Brier Score` : Deep Ensemble est le meilleur (0.02 vs 0.04 pour les autres) → Cela signifie que ses prédictions sont plus proches de la réalité en probabilité.

Concernant le Reliability Diagram :
Single et MC Dropout sont légèrement sur-confiants (les barres sont en dessous de la diagonale).

        Mais Deep Ensemble suit mieux la diagonale → meilleure correspondance entre confiance et vérité.


### **Conclusion : Reliability Diagram**
        Même si tous les modèles montrent une calibration correcte en apparence, les diagrammes de fiabilité révèlent que Deep Ensemble offre une meilleure répartition entre confiance et exactitude.

        Il est moins sur-confiant, ajuste mieux sa probabilité, et sa courbe est plus proche de la ligne idéale.
        Ce modèle se rapproche le plus du comportement qu’on attend dans un contexte médical : il est sûr quand il faut, et prudent quand il doute.

# 🔵 SLIDE : Conformal Prediction – RAPS / SAPS / APS / GAPS

J'ai ensuite appliqué différentes méthodes de prédiction conforme sur chaque modèle pour générer des ensembles de classes, au lieu d’une seule prédiction.

Concernant la Taille des ensembles : on remarque des ensembles compacts (autour de 1 à 2 labels).

Pour la couverture : Toujours proche de 0.9 – 1.0 (comme attendu).

GAPS (en haut) est le plus efficace → donne de bons ensembles avec peu d’erreurs.
On peut voir que ...

Pour Deep Ensemble + GAPS : ensemble compact + couverture parfaite + peu de violations.


### **Conclusion: PREDICTION CONFORME (set size & coverage)**
        

        on voit que plus l’entropie augmente – donc plus le modèle est incertain – plus la taille de  l'ensemble prédictif est elargie.

        SAPS, GAPS, etc .. permettent des ensembles plus compacts tout en gardant une bonne couverture.

        Pour les trois approches (SN, DE, MCD)
        On a pu observé que la taille des ensembles prédictifs varie avec les méthodes :
        Mais c’est avec Deep Ensemble + GAPS que nous obtenons le meilleur compromis entre précision, couverture et taille réduite des ensembles.

        Ce comportement est crucial en clinique : il permet au modèle de dire "je suis sûr", ou bien "voici mes deux options probables", au lieu d’inventer une fausse certitude.


"

# 🔵 SLIDE : Entropy vs Performance
🎤 Introduction :

        Ici, on observe comment l’incertitude (mesurée par l’entropie) est liée à la qualité des prédictions.


On la courbe de Rejection Plot : Plus on rejette les échantillons à forte entropie, plus l’accuracy augmente → surtout pour Deep Ensemble.

Distribution d’entropie :

Erreurs = entropie haute

Prédictions correctes = entropie très faible

        ✅ Deep Ensemble a la meilleure séparation entre correct et incorrect.



# 🔵 SLIDE : Entropy vs Set Size

"Nous avons ensuite analysé si la taille de l’ensemble prédictif s’adapte bien à l’incertitude du modèle."

Lorsque l’entropie augmente, la taille des ensembles augmente aussi.

Encore une fois, Deep Ensemble montre une meilleure corrélation entre incertitude et taille adaptative.

### **Conclusion (ENTROPY VS PREDICTION SET SIZE)**
Ici, on voit que plus l’entropie augmente – donc plus le modèle est incertain – plus il élargit naturellement la taille de son ensemble prédictif.

Ce comportement est justement ce qu’on attend d’un système intelligent :

il donne une seule réponse quand il est confiant, et plusieurs quand il doute.
Encore une fois, Deep Ensemble montre un alignement plus clair entre incertitude et ajustement dynamique.

# Conclusion
Grâce à la combinaison de méthodes probabilistes et de prédiction conforme, nous permettons au modèle :
d'exprimer son incertitude,
d’adapter la taille de ses réponses,


# 🔵 SLIDE : OOD Detection
**Analyse simple**
* Deep Ensemble est largement supérieur en AUROC (0.92) et AUPRC (0.93)
➤ Cela signifie qu’il détecte très bien les cas hors distribution (OOD) avec peu d'erreurs et une bonne précision.

* MC Dropout est meilleur que Single mais reste limité (AUROC = 0.68, AUPRC = 0.76)
➤ Il détecte mieux que Single, mais reste confus sur certains cas.

* Single Network obtient les scores les plus faibles.
➤ Cela montre qu’il a du mal à faire la différence entre des données connues et inconnues.

**✅ Conclusion à dire à l’oral**
"Deep Ensemble se distingue nettement des deux autres approches pour la détection de données inconnues.
Avec un AUROC de 0.92 et un AUPRC de 0.93, il arrive à dire quand il ne sait pas, ce qui est essentiel en contexte clinique."

