# February 24 assignment

## Implementation of SMDP-Q-learning for bottleneck options

I compare several bottleneck-options discovery mechanisms. Training is done in two phases: first, option policies are learning as well as the policy over options as to reach bottleneck states. Then the goal state is moved and the agent has to re-converge. I measure the number of steps needed to re-convergence.

Here are the mechanisms (all tabular), trained with [SMDP Q-learning](https://webdocs.cs.ualberta.ca/~sutton/papers/SPS-aij.pdf):
- Normal Q-Learning
- random sub-goals
- [bottleneck states](http://www.mcgovern-fagg.org/amy/pubs/mcgovern_barto_icml2001.pdf)
- the [graph betweenness centrality metric](https://papers.nips.cc/paper/3411-skill-characterization-based-on-betweenness.pdf)
- the [graph closeness centrality metric](http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf)
- the [graph communicability metric](https://arxiv.org/abs/0707.0756)