import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import operator as op
import pandas as pd
from functools import reduce
import random
import plotly.express as px
import altair as alt
st.title("The Condorcet Jury Theorem")

st.write('''

Consider a finite set of voters, or experts. Suppose that there are two alterantives. So the group of experts is interested in answering a question with two possible answers. Examples include: Is the defendent guilty or innocent?  Should the policy be accepted or rejected? Is the the proposition $P$ is true or false?  
''')

st.subheader("Notation")
"""* $V=\{1, 2, \ldots, n\}$ is the set of experts.
* $\{0, 1\}$ is the set of outcomes.
* $\mathbf{x}$ be a random variable (called the **state**) whose values range over the two outcomes.   We write $\mathbf{x}=1$ when the outcome is $1$ and  $\mathbf{x}=0$ when the outcome is $0$.
* $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ are random variables represeting the votes for individuals $1, 2, \ldots, n$.   For each $i=1, \ldots, n$, we  write $\mathbf{v}_i=1$ when expert $i$'s vote  is $1$ and  $\mathbf{v}_i=0$ when expert $i$'s vote  is $0$.
* $R_i$ is the event that expert $i$ votes correctly: it is  the event that $\mathbf{v}_i$ coincides with $\mathbf{x}$ (i.e., $\mathbf{v}_i=1$ and $\mathbf{x}=1$ or $\mathbf{v}_i=0$ and $\mathbf{x}=0$). 
"""

st.subheader("Competence")

"""Let $Pr(R_i)$ be the probability of the event $R_i$, called the **competence** of expert $i$.  That is, $Pr(R_i)$ is the probability that expert $i$'s vote matches the state.
"""

st.subheader("Probability that the majority is correct")

""" Suppose that the competence of each expert is $p$ (i.e., for all $i=1, \ldots, n$, $Pr(R_i)=p$).   The probability that at least $m$ experts out of $n$ being  correct is:
"""

st.latex(r'\sum_{h=m}^n \frac{n!}{h!(n-h)!} * p^h*(1-p)^{n-h}')


"""

* $\\frac{n!}{h!(n-h)!}$ is the number of ways to choose $h$ experts out of $n$
* $p^h*(1-p)^{n-h}$ is the probability that $h$ experts are correct and $n-h$ experts are incorrect. 

An important assumption needed to justify the above formula is that the event that any expert is correct is **independent** of any other expert being correct. 

For example, suppose that $V=\{1,2,3\}$ and for each $i\in V$, $Pr(R_i)=\\frac{2}{3}$. Then the probability that at least two experts are correct is the sum of the following: 

* The probability that experts 1 and 2 are correct and 3 is incorrect is: $\\frac{2}{3} * \\frac{2}{3} * \\frac{1}{3}$
* The probability that experts 2 and 3 are correct and 1 is incorrect is: $\\frac{1}{3} * \\frac{2}{3} * \\frac{2}{3}$
* The probability that experts 1 and 3 are correct and 2 is incorrect is: $\\frac{2}{3} * \\frac{1}{3} * \\frac{2}{3}$
* The probability that experts 1, 2 and 3 are correct is: $\\frac{2}{3} * \\frac{2}{3} * \\frac{2}{3}$

That is, the probability that at least 2 experts are correct is:
$3 * \\frac{2}{3} * \\frac{2}{3} * \\frac{1}{3} + \\frac{2}{3}  * \\frac{2}{3}  * \\frac{2}{3}$.
"""

"""
The following graph displays the probability that the majority is correct
"""
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return float(numer//denom)

def probability_majority_is_correct(num_voters=100,prob=0.51):
    return sum([ncr(num_voters,k)*(prob**k)*(1-prob)**(num_voters-k) 
                for k in range(int(num_voters/2+1),num_voters+1)])

@st.cache
def gen_graph_data(num_experts):
    data_for_df = {
    "Number of experts": list(),
    "Expert competence": list(),
    "Probability that the majority is correct": list(),
    }
    competeneces = np.linspace(0,1,num=100)
    for ne in num_experts: 
        for c in competeneces: 
            data_for_df["Number of experts"].append(ne)
            data_for_df["Expert competence"].append(c)
            data_for_df["Probability that the majority is correct"].append(probability_majority_is_correct(num_voters = ne, prob=c))
    return data_for_df
data_for_df =  gen_graph_data([1, 3, 11, 51,  201, 501, 1001])
df = pd.DataFrame(data_for_df)
fig = px.line(df, x="Expert competence", y="Probability that the majority is correct", color="Number of experts") 
st.plotly_chart(fig, use_container_width=True)

st.subheader("The Theorem")

"""
**Independence**: The correctness events $R_1, R_2, \ldots, R_n$ are independent.

**Competence**: The experts' competences $Pr(R_i)$  (i) exceeds $\\frac{1}{2}$ and (ii) is the same for each expert $i$.

**Condorcet Jury Theorem**. Assume Independence and Competence.  Then, as the group size increases, the probability of that the majority is correct (i) increases (growing reliability), and (ii) tends to one (infallibility).

The Condorcet Jury Theorem has two main theses: 
    
**The growing-reliability thesis**: Larger groups are better truth-trackers. That
is, they are more likely to select the correct alternative (by majority) than
smaller groups or single individuals.

**The infallibility thesis**: Huge groups are infallible truth-trackers. That is, the
likelihood of a correct (majority) decision tends to full certainty as the group
becomes larger and larger.
"""

st.subheader("Different competences")
"""
The **expert rule** chooses and expert from the group at random, then asks the expert to give decide the outcome for the group. 

The probability that the expert rule is correct is the **expected competence** of the group: $\sum_{i\in V} \\frac{1}{n}Pr(R_i)$ where $n$ is the number of elements in $V$. 

A natural question is which rule is better: The expert rule or the majority rule? 

Suppose that there are 3 experts each with possibly different competences.

"""

st.write(" ")
st.write(" ")
col1, col2  = st.columns(2)

with col1: 
    p1 = st.slider('Competence for expert 1', min_value=0.5, max_value=1.0, value=0.55, step=0.01)
    p2 = st.slider('Competence for expert 2', min_value=0.5, max_value=1.0, value=0.6, step=0.01)
    p3 = st.slider('Competence for expert 3', min_value=0.5, max_value=1.0, value=0.75, step=0.01)
with col2:
    maj_prob = p1*p2*p3 + p1*p2*(1-p3) + + p2*p3*(1-p1) + + p1*p3*(1-p2) 
    expert_prob = 1.0/3.0 * p1 + 1.0/3.0 * p2 + 1.0/3.0 * p3
    st.write(" ")
    st.write(" ")
    if maj_prob > expert_prob: 
        st.write(f"**Majority rule is better than the expert rule.**")
        st.info(f"The probability that the majority rule is correct is **{round(maj_prob,4)}**.")
        st.warning(f"The probability that the expert rule is correct is **{round(expert_prob,4)}**.")
    elif maj_prob == expert_prob: 
        st.write(f"**The expert rule and majority rule are tied.**")
        st.info(f"The probability that the majority rule is correct is **{round(maj_prob,4)}**.")
        st.info(f"The probability that the expert rule is correct is **{round(expert_prob,4)}**.")
    else:
        st.write(f"**The expert rule is better than majority rule.**")
        st.warning(f"The probability that the majority rule is correct is **{round(maj_prob,4)}**.")
        st.info(f"The probability that the expert rule is correct is **{round(expert_prob,4)}**.")
st.write("")
st.write("**Theorem**.  For any (odd) number of experts, each with a competence of $1 > p>1/2$, then the  majority rule is a greater probability of being correct than the expert rule. ")

st.write("**Theorem**.  Assume that $p_1\ge p_2>p_3>1/2$, then the  majority rule is a greater probability of being correct than the expert rule. ")

class Agent():
    
    def __init__(self, comp=0.501):
        self.comp = comp
        
    def vote(self, ev):
        #vote on whether the event is true or false
        #need the actual truth value in order to know which direction to be biased
        if ev:
            #ev is true
            return int(random.random() < self.comp)
        else:
            return 1 - int(random.random() < self.comp)


def maj_vote(the_votes):
    votes_true = len([v for v in the_votes if v == 1])
    votes_false = len([v for v in the_votes if v == 0])

    if votes_true > votes_false:
        return 1
    elif votes_false > votes_true:
        return 0
    else:
        return -1  #tied

def generate_competences(n, mu=0.51, sigma=0.2):
    competences = list()
    for i in range(0,n):
        #sample a comp until you find one between 0 and 1
        comp=np.random.normal(mu, sigma)
        while comp > 1.0 or comp < 0.0:
            comp=np.random.normal(mu, sigma)
        competences.append(comp)
    return competences


st.subheader("Competence")

"""
The assumption that the experts' competence is greater than 0.5 can be weakened.  Select the maximum number of  experts, an average competenece for the experts and the standard deviation of the competences.   Then, the graph shows the probability that the majority is correct compared to a the expert rule (randomly selecting an expert that selects an outcome) for different numbers of agents (up to the maximum number of experts). 
"""
st.write("")
col1, col2 = st.columns(2)
P=True

with col1: 
    max_num_voters = st.slider("Maximum number of experts", min_value=3, max_value=201, value=101, step=2)
    total_num_voters = range(1,max_num_voters)

    comp_mu = st.slider("Average Competence", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    comp_sigma = st.slider("Standard deviation of the competences", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
with col2: 
    competences = generate_competences(max_num_voters,
                                       mu=comp_mu, 
                                       sigma=comp_sigma)

    df = pd.DataFrame({"Expert": range(len(competences)), "Competence": competences})
    c = alt.Chart(df, title="Expert Competences").mark_circle(size=60).encode(x='Expert', y='Competence', tooltip=['Competence'])
    st.altair_chart(c, use_container_width=True)
    #if comp_sigma > 0:  
    #    fig = px.scatter(df, x="Expert", y="Competence", trendline="ols", #title="Competences")
    #else: 
    #    fig = px.scatter(df, x="Expert", y="Competence", title="Competences")

NUM_ROUNDS = 500
maj_probs = list()
expert_probs = list()

for num_idx, num_voters in enumerate(total_num_voters):

    experts = [Agent(comp=competences[num-1]) for num in range(0,num_voters)]
    
    maj_votes = list()
    expert_votes = list()
    for r in range(0,NUM_ROUNDS):
        votes = [a.vote(P) for a in experts]
        maj_votes.append(maj_vote(votes))
        
        expert_votes.append(random.choice(experts).vote(P))
    
    maj_probs.append(float(float(len([v for v in maj_votes if v==1]))/float(len(maj_votes))))
    expert_probs.append(float(len([v for v in expert_votes if v==1]))/float(len(expert_votes)))

data_for_df = {"Number of Experts": list(), "Probability correct": list(), "Rule": list()}

for ne_idx, ne in enumerate(total_num_voters): 
    data_for_df["Number of Experts"].append(ne)
    data_for_df["Probability correct"].append(maj_probs[ne_idx])
    data_for_df["Rule"].append("Majority")
    data_for_df["Number of Experts"].append(ne)
    data_for_df["Probability correct"].append(expert_probs[ne_idx])
    data_for_df["Rule"].append("Expert")

df = pd.DataFrame(data_for_df)
fig = px.line(df, x="Number of Experts", y="Probability correct", color="Rule") 
fig.update_layout(yaxis_range=[0,1])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Independence")
"""
The following example illustrates the importance of the assumption that the experts opinions are **independent**. 

Suppose that there are 3 experts $V=\{1,2,3\}$.  Consider a probability over the joint space specifying the votes of each expert.   The outcomes are the 8 possible voting scenarios 

1. $\mathbf{v}_1=1, \mathbf{v}_2=1, \mathbf{v}_3=1$
2. $\mathbf{v}_1=1, \mathbf{v}_2=1, \mathbf{v}_3=0$
3. $\mathbf{v}_1=1, \mathbf{v}_2=0, \mathbf{v}_3=1$
4. $\mathbf{v}_1=1, \mathbf{v}_2=0, \mathbf{v}_3=0$
5. $\mathbf{v}_1=0, \mathbf{v}_2=1, \mathbf{v}_3=1$
6. $\mathbf{v}_1=0, \mathbf{v}_2=1, \mathbf{v}_3=0$
7. $\mathbf{v}_1=0, \mathbf{v}_2=0, \mathbf{v}_3=1$
8. $\mathbf{v}_1=0, \mathbf{v}_2=0, \mathbf{v}_3=0$

Suppose that that correct outcome is 1.   Let $p_i$ be the probability of voting situation $i$.  So, for instance, $p_4=0.25$ means that there is a probability of $0.25$ that expert 1 votes $1$, expert 2 votes $0$ and expert 3 votes $0$.  Then, given a probability over the 8 possible voting situations (a sequence of 8 numbers $(p_1,\ldots, p_i)$ that are nonnegative and sum to 1), we have the following: 

* the probability that expert 1 votes correctly is $p_1+p_2+p_3+p_4$.
* the probability that expert 2 votes correctly is $p_1+p_2+p_4+p_5$.
* the probability that expert 3 votes correctly is $p_1+p_3+p_5+p_7$.
* the probability that a majority is correct is $p_1 + p_2 + p_3 + p_5$

The graph below shows the probability that the majority is correct compared to the expert rule (the probability that a randomly selected expert is correct) for randomly generated probabilities over the 8 possible voting situations.  

The experts are independent if the product of the probability that expert 1 is correct,  the probability that expert 2 is correct and the probability that expert 3 is correct equals $p_1$. I.e., the experts are indpendent when $(p_1+p_2+p_3+p_4)*(p_1+p_2+p_4+p_5)*(p_1+p_3+p_5+p_7) = p_1$.  Note that it is very unlikely that the experts votes will be independent. 
"""

def gen_options_probability_dirichlet(params, num=100):
    return np.random.dirichlet(params, num)

def gen_data(n): 
    prs = gen_options_probability_dirichlet((1,1,1,1,1,1,1,1), num=n)
    maj_preferred_expert = 0
    data_for_df= {"N": list(), "Probability correct":list(), "Rule": list()}
    num_independent = 0
    for pr_idx,pr in enumerate(list(prs)): 
        pr1 = pr[0] + pr[1] + pr[2] + pr[3]
        pr2 = pr[0] + pr[1] + pr[4] + pr[5]
        pr3 = pr[0] + pr[2] + pr[4] + pr[6]
        
        num_independent += int((pr1 * pr2 * pr3) == pr1)
        
        maj_pr_correct = pr[0] + pr[1] + pr[2] + pr[4]
        data_for_df["N"].append(f"{pr_idx}")
        data_for_df["Probability correct"].append(maj_pr_correct)
        data_for_df["Rule"].append("Majority")
        expert_pr_correct = np.average([pr1, pr2, pr3])
        data_for_df["N"].append(f"{pr_idx}")
        data_for_df["Probability correct"].append(expert_pr_correct)
        data_for_df["Rule"].append("Expert")

    return data_for_df, num_independent

num_prs = st.slider("Number of probabilities", min_value=5, max_value=200, value=100, step=1)

data_for_df, num_independent = gen_data(num_prs)
st.write(f"In the following graph, the number of probabilities in which the experts are indpendent is {num_independent}.")
df = pd.DataFrame(data_for_df)
fig = px.line(df, x="N", y="Probability correct", color="Rule") 
fig.update_layout(yaxis_range=[0,1])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Further Reading")

## Further Reading 
"""
D. Austen-Smith and J. Banks, Aggregation, Rationality and the Condorcet Jury Theorem, The American Political Science Review, 90, 1, pgs. 34 - 45, 1996
 
F. Dietrich, The premises of Condorcet's Jury Theorem are not simultaneously justified, Episteme, Episteme - a Journal of Social Epistemology 5(1): 56-73, 2008

F. Dietrich and K. Spiekermann (2020). [Jury Theorems: a review](http://www.franzdietrich.net/Papers/DietrichSpiekermann-JuryTheorems.pdf),  In: M. Fricker et al. (eds.) The Routledge Handbook of Social Epistemology. New York and Abingdon: Routledge

D. Estlund, Opinion Leaders, Independence and Condorcet's Jury Theorem, Theory and Decision, 36, pgs. 131 - 162, 1994

R. Goodin and K. Spiekermann, *An Epistemic Theory of Democracy*, Oxford University Press, 2018

U. Hahn, M. von Sydow and C. Merdes (2018). [How Communication Can Make Voters Choose Less Well](https://onlinelibrary.wiley.com/doi/epdf/10.1111/tops.12401), Topics in Cognitive Science

C. List and R. Goodin. Epistemic democracy: Generalizing the Condorcet Jury Theorem. Journal of Political Philosophy, 9(3):277–306, 2001.

"""