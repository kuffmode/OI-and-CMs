# When There Is No Better Solution: *Characterizing Optimal Signal Propagation in the Human Brain Network.*
## Authors:
- Kayson Fakhar<sup>*1</sup>
- Fatemeh Hadaeghi<sup>1</sup>
- Caio Seguin<sup>2</sup>
- Shrey Dixit<sup>1</sup>
- Arnaud Messé<sup>1</sup>
<<<<<<< HEAD
- Gorka Zamora-Lopez<sup>3,4</sup>
- Bratislav Misic<sup>5</sup>
- Claus C. Hilgetag<sup>1,6</sup>
=======
- Bratislav Misic<sup>3</sup>
- Claus C. Hilgetag<sup>1,4</sup>
>>>>>>> origin/main

<sub>
1. Institute of Computational Neuroscience, University Medical Center Eppendorf-Hamburg, Hamburg University, Hamburg Center of Neuroscience, Germany.
2. Department of Psychological and Brain Sciences, Indiana University, Bloomington, IN, USA.
<<<<<<< HEAD
3. Center for Brain and Cognition, Pompeu Fabra University, 08005 Barcelona, Spain.
4. Department of Information and Communication Technologies, Pompeu Fabra University, 08018 Barcelona, Spain.
5. McConnell Brain Imaging Centre, Montréal Neurological Institute, McGill University, Montréal, Canada.
6. Department of Health Sciences, Boston University, Boston, MA, USA.
=======
3. McConnell Brain Imaging Centre, Montréal Neurological Institute, McGill University, Montréal, Canada.
4. Department of Health Sciences, Boston University, Boston, MA, USA.
>>>>>>> origin/main
* kayson.fakhar@gmail.com
</sub>


## Abstract:
Communication within large-scale brain networks is the cornerstone of cognitive function and behavior. It is hypothesized that a multitude of evolutionary pressures, including the minimization of wiring costs and the optimization of signaling efficiency, contribute to shaping how regions interact. Although various interpretations of ‘optimal’ signaling were previously provided, namely, directing information over the path with the fewest number of intermediate nodes, a mathematically rigorous definition as well as its characteristics was lacking. How would the landscape of neural communication look like if regions were to maximize their influence over each other in a given brain network model?

Our study answers this question by combining structural data from human cortical networks, computational models of brain dynamics, and an assignment of influences among the nodes based on game theory. We quantified the exact influence exerted by each node over every other node, using a game-theoretical framework relying on an exhaustive multi-site in-silico lesioning scheme, creating  optimal influence maps for various brain network models. These descriptions show how signaling should unfold in the given brain network if regions were to maximize their influence over one another. Next, by comparing these maps against alternative brain communication models, we found the optimal communication to differ from the traditional view in network neuroscience, in which signaling is confined to the shortest paths between nodes. Instead, it resembles a broadcasting system, leveraging multiple parallel channels for information dissemination. Furthermore, our investigation revealed that the most influential regions within the cortex are its well-known rich-club. These regions exploit their topological vantage points, broadcasting across numerous pathways, thus significantly enhancing their effective reach even when the anatomical connections are weak. 

Altogether, our work firstly provides a rigorous and versatile framework to define and delineate optimal signaling across networks, with a particular lens on the brain. It then underscores which regions are the most influential, and what topological features allow them to be influential. 

## How this repository is organized:
Quick answer is that it's not really organized. But to do the simulations you can use ```simulations_linear.py``` or if you have some nice computers with many CPU cores (for us N=512, and 256 GB of RAM, still took three weeks) then you can try ```simulations_hopf_model.py```. HOWEVER, we thought about this and wrote a separate nifty library for this purpose called [YANAT: Yet Another Network Analysis Toolkit](https://github.com/kuffmode/YANAT).

Anyway, you can see how things were done after the simulations in the notebook ```OI and CMs.ipynb```, see some extra comparisons in ```model_comparison.ipynb``` and the fitting process in ```fitting_dynamics.ipynb```. All of them are poorly organized just because I didn't have time to do it properly at the moment (look man this already took 2 years, look at the first commit, I have to finish my PhD at some point).

There are some surface plots using a MATLAB package called [Simple brain plot](https://github.com/dutchconnectomelab/Simple-Brain-Plot) from the Dutch Connectome Lab. You can see the code in ```visualization.m```. I love this package but I gotta warn you it took me a while to figure out how to mix it with my Python stuff.

Lastly, to have nice (and visually inclusive) plots I used my own specifications and typeface that you can find in ```visual_config.py```. I will have it also separated in the future in a GitHub repository but for now, you can just copy and paste it in your Python script.