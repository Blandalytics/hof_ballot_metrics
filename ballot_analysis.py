import streamlit as st
from streamlit import session_state as ss

import matplotlib.colors as colors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import urllib

from PIL import Image
from pyfonts import set_default_font, load_google_font
from scipy import stats


st.set_page_config(page_title='Baseball Hall of Fame Ballot Analysis', page_icon='⚾',
                   layout="wide"
                  )
# For logs
pd.set_option('future.no_silent_downcasting', True)

new_title = '<p style="color:#72CBFD; font-weight: bold; font-size: 30px;">Baseball Hall of Fame Ballot Analysis</p>'
st.markdown(new_title, unsafe_allow_html=True)
st.write('To change voters/years, tap the >> in the upper left of the page')

def load_logo():
    logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    img_url = urllib.request.urlopen(logo_loc)
    logo = Image.open(img_url)
    return logo

logo = load_logo()

def letter_logo():
    logo_loc = 'https://github.com/Blandalytics/baseball_snippets/blob/main/teal_letter_logo.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

letter_logo = letter_logo()

font = load_google_font("Alexandria")
fm.fontManager.addfont(str(font.get_file()))

## Set Styling
# Plot Style
pl_white = '#FFFFFF'
pl_background = '#292C42'
pl_text = '#72CBFD'
pl_line_color = '#8D96B3'
pl_highlight = '#F1C647'
pl_highlight_gradient = ['#F1C647','#F5A05E']
pl_highlight_cmap = sns.color_palette(f'blend:{pl_highlight_gradient[0]},{pl_highlight_gradient[1]}', as_cmap=True)

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_line_color,
        'ytick.color': pl_line_color,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     },
    font='Alexandria'
    )

gid_dict = {
    2026:'0',
    2025:'922799909',
    2024:'2131068923',
    2023:'1282570088',
    2022:'903695893',
    2021:'408912314',
    2020:'340817630',
    2019:'1014647114',
    2018:'293466111',
    2017:'390426574',
}

update_text = pd.read_csv(f'https://docs.google.com/spreadsheets/d/1i7ZOAxOPnpCkCiiJ-bxnjWZ2vxFhBThz4C70ZpOAhW8/export?gid=0&format=csv').fillna(0).columns.values[0]
update_text = update_text[update_text.find('\n')+1:update_text.find(' at')]

loading_texts = [
    "Figuring out why Ichiro wasn't unanimous",
    'Determining which voter hates your team the most',
    'Finding the optimal balance between eye test and numbers',
    'Sorting by word count in ballot announcements',
    'Finding the optimal amount of "Fame" for a Hall with set dimensions'
]

@st.cache_data(ttl=3600, show_spinner=random.choice(loading_texts))
def load_data(sheet_id_dict):
    year_min = min(gid_dict.keys())
    year_max = max(gid_dict.keys())
    dfs_list = []
    for year in gid_dict.keys():
        gid = gid_dict[year]
        tracker_year = pd.read_csv(f'https://docs.google.com/spreadsheets/d/1i7ZOAxOPnpCkCiiJ-bxnjWZ2vxFhBThz4C70ZpOAhW8/export?gid={gid}&format=csv').fillna(0)
        nominees = [x.replace('\n',' ').replace('  ',' ').replace('- ','')[:-6].strip()+f'_{year}' for x in tracker_year.columns.values][1:]
        tracker_year.columns = ['Voter'] + nominees
        tracker_year[nominees] = tracker_year[nominees].replace('x',1)
        tracker_year = tracker_year.loc[tracker_year['Voter']!=0].reset_index(drop=True)
        tracker_year['Voter'] = tracker_year['Voter'].str.replace(r'\(([^)]+)\)', '', regex=True).str.replace('\n','').str.replace('*','').str.strip()
        type_dict = {x:'int' for x in nominees}
        type_dict.update({'Voter':'str'})
        tracker_year = tracker_year.astype(type_dict)
        tracker_year['Total Votes'] = tracker_year[nominees].sum(axis=1)

        # Stinginess is just Number of votes used, belowe average
        tracker_year['raw_vote_stinginess'] = tracker_year[nominees].sum(axis=1).mean() - tracker_year['Total Votes']

        # Orthodoxy is MAE of a voter's ballot, relative to others that year
        for player in nominees:
            tracker_year[player+'_avg'] = tracker_year[player].astype('int')
            tracker_year[player] = tracker_year[player] - tracker_year[player].mean()
        tracker_year['raw_vote_orthodoxy'] = tracker_year[nominees].abs().mean(axis=1)
        tracker_year['votes_orth_max'] = tracker_year['raw_vote_orthodoxy'].max()
        tracker_year['votes_orth_min'] = tracker_year['raw_vote_orthodoxy'].min()
        tracker_year['vote_orthodoxy'] = tracker_year['votes_orth_max'].sub(tracker_year['raw_vote_orthodoxy']).div(tracker_year['votes_orth_max'].sub(tracker_year['votes_orth_min'])).mul(100)

        # "Adjusted" Orthodoxy is MAE, within the cohort of voters who have cast the same number of votes
        tracker_year['votes_orth_max_cohort'] = tracker_year['raw_vote_orthodoxy'].groupby(tracker_year['Total Votes']).transform('max')
        tracker_year['votes_orth_min_cohort'] = tracker_year['raw_vote_orthodoxy'].groupby(tracker_year['Total Votes']).transform('min')
        tracker_year['vote_orthodoxy_cohort'] = tracker_year['votes_orth_max_cohort'].sub(tracker_year['raw_vote_orthodoxy']).div(tracker_year['votes_orth_max_cohort'].sub(tracker_year['votes_orth_min_cohort'])).mul(100)
    
        dfs_list += [tracker_year.assign(year=year)]
    
    tracker_years = pd.concat(dfs_list,ignore_index=True).sort_values(['Voter','year']).reset_index(drop=True)
    tracker_years['vote_stinginess'] = tracker_years['raw_vote_stinginess'].sub(tracker_years['raw_vote_stinginess'].min()).div(tracker_years['raw_vote_stinginess'].max() - tracker_years['raw_vote_stinginess'].min()).mul(100)
    tracker_years['stinginess_y+1'] = np.where((tracker_years['Voter']==tracker_years['Voter'].shift(-1)) &
                                           (tracker_years['year'].add(1)==tracker_years['year'].shift(-1)),
                                           tracker_years['vote_stinginess'].shift(-1),
                                           None)
    tracker_years['orthodoxy_y+1'] = np.where((tracker_years['Voter']==tracker_years['Voter'].shift(-1)) &
                                           (tracker_years['year'].add(1)==tracker_years['year'].shift(-1)),
                                           tracker_years['vote_orthodoxy'].shift(-1),
                                           None)
    tracker_years['first_time_voter'] = np.where((tracker_years['year'].groupby(tracker_years['Voter']).transform('max')==year_max) & (tracker_years['year'].groupby(tracker_years['Voter']).transform('count')==1),1,0)
    return tracker_years

tracker_years = load_data(gid_dict)

# Intialize session state
if 'year' not in ss:
    ss['year'] = max(gid_dict.keys())
if 'voter' not in ss:
    ss['voter'] = tracker_years.loc[tracker_years['year']==ss['year'],'Voter'].sample(1).item()

with st.sidebar:
    pad1, col1, pad2 = st.columns([0.2,0.6,0.2])
    with col1:
        st.image(letter_logo)
    voter_list = tracker_years['Voter'].sort_values().unique()
    st.selectbox('Select a voter',
                 voter_list,
                 key='voter')
  
    years_list = tracker_years.loc[tracker_years['Voter']==ss['voter'],'year'].sort_values(ascending=False).unique()
    st.selectbox('Select a ballot year',
                 years_list,
                 key='year')

player_options = [x for x in tracker_years.loc[tracker_years['year']==ss['year']].columns.values[1:-4] if x[-5:]==f'_{ss['year']}']
voted_players = [x for x in tracker_years.loc[tracker_years['year']==ss['year']].columns.values[1:-4] if f'_{ss['year']}_avg' in x]
unanimous_players = [x[:-9] for x in voted_players if tracker_years[x].mean()==1]

def ballot_chart(voter, year):
    voter_df = tracker_years.loc[(tracker_years['Voter']==voter) & (tracker_years['year']==year)].reset_index(drop=True)
    chart_df = voter_df[player_options].loc[:, (voter_df[player_options].abs() > 0.05).all()].T.reset_index().assign(Player = lambda x: x['index'].str[:-5]).rename(columns={0:'Votes Above Average'})
    
    fig = plt.figure(figsize=(10,6))
    # # Divide card into tiles
    grid = plt.GridSpec(3, 2,hspace=5,wspace=0,width_ratios=[3,2])
    # fig, ax = plt.subplots(figsize=(8,6))
    ax1 = plt.subplot(grid[:, 0])
    hue_norm = colors.CenteredNorm(0,1)
    sns.barplot(chart_df.sort_values('Votes Above Average',ascending=False),
                y='Player',
                x='Votes Above Average',
                hue='Votes Above Average',
                hue_norm=hue_norm,
                palette='PuOr_r',
                # color=pl_line_color,
                edgecolor=pl_line_color,
                ax=ax1,
                legend=False)
    ax1.axvline(0,color=pl_line_color)
    ax1.axhline(-0.7,color=pl_line_color,xmin=1/8,xmax=7/8,linewidth=2)
    player_votes = voter_df['Total Votes'].item()
    asterisk_text = '' if len(unanimous_players)==0 else '*'
    ax1.text(1.5,-1,f'Voted For ({player_votes:.0f}{asterisk_text})',ha='right',color=pl_line_color)
    ax1.text(-1.5,-1,'Did Not Vote For',ha='left',color=pl_line_color)
    if len(unanimous_players)>0:
        ax1.text(0.25,chart_df.shape[0]-2,
                 '*Unanimous Players\n(Public Ballots):\n- '+'- \n'.join(unanimous_players),
                 color=pl_line_color,alpha=0.75)
    for player in ax1.get_yticklabels():
        vaa = chart_df.loc[chart_df['Player']==player.get_text(),'Votes Above Average'].item()
        ax1.text(vaa + (0.02 if vaa >0 else -0.02),
                 ax1.get_yticklabels().index(player),
                 player.get_text(),
                 # fontsize=10,
                 va='center',
                 ha='left' if vaa >0 else 'right',
                 color='w')
    ax1.yaxis.set_visible(False)
    ax1.set_xticks([-1,-0.5,0,0.5,1])
    ax1.set(xlabel='',
           xlim=(-2,2),
           ylabel='')
    ax1.set_title("Votes Above Average",color=pl_text,fontsize=16,y=1.01)
    
    x_adjust = 115
    
    ax2 = plt.subplot(grid[0, 1])
    sns.kdeplot(tracker_years['vote_stinginess'],
                cut=0,
                bw_adjust=1.5,
                fill=True,
                color=pl_line_color,
                alpha=0.5,
                ax=ax2)
    stingy_val = voter_df['vote_stinginess'].item()
    ax2.axvline(stingy_val,color=pl_highlight,linestyle='--',linewidth=2)
    ax2.yaxis.set_visible(False)
    ax2.set_xticks([x*20 for x in range(6)])
    ax2.set(xlim=(-0.5,x_adjust),
           xlabel='')
    ax2.set_title(f'Vote Stinginess: {stingy_val:.0f}',color=pl_text,fontsize=16,x=50/x_adjust,y=1.01)
    sns.despine(left=True)
    
    ax3 = plt.subplot(grid[1, 1])
    sns.kdeplot(tracker_years['vote_orthodoxy'],
                cut=0,
                bw_adjust=1.5,
                fill=True,
                color=pl_line_color,
                alpha=0.5,
                ax=ax3)
    orth_val = voter_df['vote_orthodoxy'].item()
    ax3.axvline(orth_val,color=pl_highlight,linestyle='--',linewidth=2)
    ax3.yaxis.set_visible(False)
    ax3.set_xticks([x*20 for x in range(6)])
    ax3.set(xlim=(-0.5,x_adjust),
           xlabel='')
    ax3.set_title(f'Vote Orthodoxy: {orth_val:.0f}',color=pl_text,fontsize=16,x=50/x_adjust,y=1.01)
    
    ax4 = plt.subplot(grid[2, 1])
    sns.kdeplot(tracker_years['vote_orthodoxy_cohort'],
                cut=0,
                bw_adjust=1.5,
                fill=True,
                color=pl_line_color,
                alpha=0.5,
                ax=ax4)
    orth_adj_val = voter_df['vote_orthodoxy_cohort'].item()
    ax4.axvline(orth_adj_val,color=pl_highlight,linestyle='--',linewidth=2)
    ax4.yaxis.set_visible(False)
    ax4.set_xticks([x*20 for x in range(6)])
    ax4.set(xlim=(-0.5,x_adjust),
           xlabel='')
    ax4.set_title(f'Vote Orthodoxy (Adjusted): {orth_adj_val:.0f}',color=pl_text,fontsize=16,x=50/x_adjust,y=1.01)
    
    # Add PL logo
    pl_ax = fig.add_axes([0.4,-.1,0.2,0.2], anchor='SE', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    
    fig.text(0.8,-0.025,'Data: Ryan Thibodaux\nwww.tracker.fyi',fontsize=10,color=pl_line_color,ha='center',va='center')
    fig.text(0.25,-0.025,f'bbhof-ballot-metrics.streamlit.app\nLast Updated: {update_text}',fontsize=10,color=pl_line_color,ha='center',va='center')
    
    fig.suptitle(f"{voter}'s {year} HoF Ballot Metrics",fontsize=20,color=pl_highlight)
    sns.despine(left=True,bottom=True)
    grid.tight_layout(fig)
    st.pyplot(fig)

pad1, col1, pad2 = st.columns([0.01,1,0.01],width=700)
with col1:
    ballot_chart(ss['voter'], ss['year'])
st.markdown("""
    - :primary[**Votes Above Average**]: Voter's decision to vote for a player (1) or not vote for a player (0), minus the percent of ballots that player was on.
    - :primary[**Vote Stinginess**]: Average # of players voted that year - the # of players voted for by that voter that year. Scaled 0 (Voted for the most players, relative to a year's average) to 100 (Voted for the fewest players, relative to a year's average).
    - :primary[**Vote Orthodoxy**]: The average distance between a voter's ballot and the average of all ballots that year. Scaled 0 (Furthest from the average) to 100 (Closest to the average).
    - :primary[**Vote Orthodoxy (Adjusted)**]: The average distance between a voter's ballot and the average of ballots that voted for the same number of players that the voter did. Scaled 0 (Furthest from the average) to 100 (Closest to the average).
    """)
