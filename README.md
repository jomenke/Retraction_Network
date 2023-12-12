# Retraction Network Dynamics
This project aims to implement and extend the prior work found in "The Dynamics of Retraction in Epistemic Networks" (LaCroix et al., 2021). Their code is available here: https://github.com/ageil/fakenews. The overarching goal is to model the spread of misinformation in networks, specifically information from retracted research articles within the scientific literature.

## Introduction
In scientific research, publications are sometimes retracted. This process usually involves adding a flag or some other indicator to the original article and publishing a notice of retraction. In general, manuscripts are retracted when an error is detected that casts doubt over the data or findings. More specifically, retractions can occur as a result of scientific misconduct, unintentional errors, plagarism, or violating ethical guidelines. Retractions are relatively rare, occuring roughly 4 times out of every 10,000 papers (Brainard and You, 2018). There are more retractions today (1000 per year in 2014) than in the past (100 per year before 2000), although this is most likely due to absolute increases in the volume of publications and better policing rather any increase in fraud (Brainard and You, 2018). Overall, retracted research should be avoided unless you are intimately familiar with the reasons for retraction (or your work focuses on retractions). If not, there is a risk of spreading misinformation. For example, the MMR vaccine was incorrectly linked to autism in a now infamous, retracted paper (Wakefield et al., 1998). And while this original paper was eventually retracted, its tainted influence still lingers even through today. Unfortunately, retracted works continue to be cited and only 5.4% of these citations reference the retraction (Hsiao and Schneider, 2021). 

Prior work by LaCroix and colleagues modeled researchers within social networks, spreading ideas amongst each other. The authors modeled this spread over a variety of network types including fully connected networks, small world networks (Watts and Strogatz, 1998), and preferential-attachment networks (Barabási and Albert, 1999). Their major findings were as follows: 

- False information often perseveres despite retractions.
- Delaying retraction can increase its effectiveness.
- Retractions are most successful when issued by the original source.

Overall, this work aims to adapt their model from researchers within social networks to research articles within citation networks. Currently, a custom network has been implemented, but adjustments are still in progress for incorporating this within the simulation. 

## Instructions
To perform simulations, adjust the run_experiment parameters as necessary and run the main.py script. 

While the previously implemented networks should still function within the simulation, the simulation involving the custom network is still being developed. The networks characteristics (i.e., average number of references and citations for each article) can be seen by running the barabasi_albert_graph_modified.py file.

## References
Brainard, Jeffrey, and Jia You. "What a massive database of retracted papers reveals about science publishing’s ‘death penalty’." Science 25.1 (2018): 1-5.

Hsiao, Tzu-Kun, and Jodi Schneider. "Continued use of retracted papers: Temporal trends in citations and (lack of) awareness of retractions shown in citation contexts in biomedicine." Quantitative Science Studies 2.4 (2021): 1144-1169.

Wakefield, Andrew J., et al. "RETRACTED: Ileal-lymphoid-nodular hyperplasia, non-specific colitis, and pervasive developmental disorder in children." The lancet 351.9103 (1998): 637-641.

LaCroix, Travis, Anders Geil, and Cailin O’Connor. "The dynamics of retraction in epistemic networks." Philosophy of Science 88.3 (2021): 415-438.

Watts, Duncan J., and Steven H. Strogatz. "Collective dynamics of ‘small-world’ networks." nature 393.6684 (1998): 440-442.

Barabási, Albert-László, and Réka Albert. "Emergence of scaling in random networks." science 286.5439 (1999): 509-512.
