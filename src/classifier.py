from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
import re

def load_path(directory, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"../{directory}/{filename}")
    model_path = os.path.normpath(model_path)
    return model_path
    
def load_classifier():
    model_path = load_path('models', 'gru_fake_news_model.keras')
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    return model

def load_word2vec():
    w2v_path = load_path('models', 'word2vec_fake_news.model')
    word2vec = Word2Vec.load(w2v_path)
    return word2vec

def load_wordindex():
    windex_path = load_path('models', 'word_index.pkl')
    with open(windex_path, "rb") as f:
        word_index = pickle.load(f)
    return word_index

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    STOPWORDS = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def tokens_to_indices(tokens, word_index):
    return [word_index.get(token, 0) for token in tokens]

def analyze_oov(tokens, word_index):
    total = len(tokens)
    unknowns = sum(1 for t in tokens if t not in word_index)
    return unknowns, total, unknowns / total


def predict_news(text):
    model = load_classifier()
    word_index = load_wordindex()
    tokens = preprocess_text(text)

    oov_count, total, oov_ratio = analyze_oov(tokens, word_index)
    print(f"OOV words: {oov_count}/{total} ({oov_ratio:.2%})")

    indices = tokens_to_indices(tokens, word_index)
    padded = pad_sequences([indices], maxlen=300, padding='post')
    pred = model.predict(padded)[0][0]
    label = "Fake" if pred >= 0.5 else "Real"
    print(f"Prediction: {label} ({pred:.4f})")
    return label, pred


if __name__ == "__main__":
    # True
    # text = 'trump twitter (dec 27) - trump, iraq, syria following statements were posted verified twitter accounts u.s. president donald trump, @realdonaldtrump @potus.  opinions expressed own. reuters edited statements confirmed accuracy.  @realdonaldtrump : - “on 1/20 - day trump inaugurated - estimated 35,000 isis fighters held approx 17,500 square miles territory iraq syria. 12/21, u.s. military estimates remaining 1,000 fighters occupy roughly 1,900 square miles...” via @jamiejmcintyre  [1749 est] - left west palm beach fire & rescue #2. met great men women representatives much us. firefighters, paramedics, first responders - amazing people are! [1811 est] - “on 1/20 - day trump inaugurated - estimated 35,000 isis fighters held approx 17,500 square miles territory iraq syria. 12/21, u.s. military est remaining 1,000 fighters occupy roughly 1,900 square miles..” @jamiejmcintyre @dcexaminer [2109 est] - ""arrests ms-13 members, associates 83% trump"" bit.ly/2lirh3b [2146 est] -- source link: (bit.ly/2jbh4lu) (bit.ly/2jpexyr) '
    text = "Ever since Donald Trump returned to power, pundits have struggled to find apt analogies for his style of governance. Some liken his loyalty demands, patronage networks and intimidation tactics to the methods of a mafia don. Others cast him as a feudal overlord, operating a personality cult rooted in charisma and bound by oaths, rewards and threats rather than laws and institutions. A growing number of artists and AI creatives are depicting him as a Viking warrior. And of course, fierce debates continue over whether the moment has arrived for serious comparisons with fascist regimes. While some of these analogies may offer a degree of insight, they are fundamentally limited by their Eurocentrism – as if 21st-century US politics must still be interpreted solely through the lens of old-world history. If we truly want to understand what is unfolding, we must move beyond Scandinavian sagas and Sicilian crime lore. I’ve found it increasingly difficult not to see striking parallels between recent events in the US and the rise of cold war-era dictatorships in Africa. It began with Trump’s renaming of the Gulf of Mexico and Denali, which recalled how Mobutu Sese Seko, on a personal whim, changed Congo into Zaire in 1971. Geographical renaming has been extensive in Africa because of its history of colonialism, but now the US has started changing names too. Trump’s deployment of national guard troops and marines to Los Angeles after protests over immigration raids also echoed Mobutu’s preferred method for dealing with civil unrest: presidential guards patrolling the streets to crush protests. The blunt use of military force to suppress domestic opposition is a tactic associated with figures such as Idi Amin in Uganda, Robert Mugabe in Zimbabwe and Paul Biya in Cameroon – albeit with deadlier consequences. Mobutu Sese Seko speaks to reporters. Mobutu Sese Seko addresses reporters outside his residence in Kinshasa, the Democratic Republic of the Congo, 23 March 1997. Photograph: Remy de la Mauvinière/AP Trump’s aggressive deportation of undocumented Latino workers also resembles Amin’s 1972 expulsion of Uganda’s Asian minority. Amin framed it as a way to return economic power to “the ordinary Ugandan”, but it led to financial ruin. The embrace of bizarre, theatrical economic measures that look great on television but wreak havoc in practice is another striking parallel. Trump’s tariffs, announced with patriotic fanfare on “liberation day”, evoke Mugabe’s grandiose land reforms of the 1980s, which hastened Zimbabwe’s collapse. Anti-intellectualism, egomania and delusions of grandeur were hallmarks of dictatorships in Africa. Ivory Coast’s Félix Houphouët-Boigny built a replica of St Peter’s Basilica in his home town. Jean-Bédel Bokassa crowned himself “emperor” of Central African Republic. “Marshal” Mobutu ensured that Concorde could land in his native village. A similar extravaganza of ambition has reached the US, with Trump accepting a luxury Boeing 747 from Qatar and hoping his face will be carved into Mount Rushmore beside George Washington, Thomas Jefferson, Theodore Roosevelt and Abraham Lincoln. The army parade in Washington on the day the US military turned 250 and Trump turned 79 was another moment of self-aggrandising narcissism. A populist personality cult and masculine pride often go hand in hand with deep paranoia and contempt. Trump’s relentless war on academia and the free press fits squarely within this tradition. In Equatorial Guinea, President Francisco Macías Nguema outlawed the word “intellectual” and prosecuted academics. Amin terrorised universities to the point of brain-drain. At first glance, viewing Trump as a westernised version of one of Africa’s dictators may seem jarring. After all, his interest in the continent appears limited to its natural resources, not its political models. The trade tariffs and travel bans he recently unleashed have hit several African countries hard, and his cruel withdrawal of aid hardly suggests admiration for anything African. What’s more, Trump has never set foot on African soil and reportedly dismissed the continent as a cluster of “shithole countries”. Only when a raw materials deal is in sight does he spring into life, such as last week when a “peace deal” between the Democratic Republic of the Congo and Rwanda was signed at the White House. “We’re getting, for the United States, a lot of the mineral rights from the Congo as part of it,” Trump said. But once the comparison between Trump and a cold war dictator is made, it becomes hard to unsee. And it shouldn’t surprise us. The postcolonial dictator was, to a significant degree, an American creation. Sooner or later, it had to come home. The US supported repressive regimes unconditionally during the cold war, viewing them as bulwarks against communism – not just in Africa, but in Asia and Latin America. Dictators such as Ferdinand Marcos in the Philippines, Suharto in Indonesia, Augusto Pinochet in Chile and Jorge Rafaél Videla in Argentina remained in power for decades thanks to US backing. When the Soviet Union collapsed, the US abruptly abandoned these allies and championed the gospel of democratisation. Though the 1990s were rich in rhetoric about human rights, good governance and the rule of law, on the ground the spectre of autocracy never vanished entirely. We’re now witnessing a startling reversal. With the demise of USAID and its retreat from a role promoting global democracy, it’s not only that the US has turned its back on democratising countries in Africa and elsewhere – but that it has begun to imitate some of the worst historical examples of authoritarian rule. Viewing Trump’s regime through the lens of cold war-era autocracies in postcolonial states offers a framework that is both alarming and oddly reassuring. If there is one enduring lesson from the history of autocracy in Africa, it is this: things can turn ugly, fast. Cold war dictatorships were ruthless, bloody and often ended in chaos and state collapse. Yet their histories also show that when courts are neutered and legislatures reduced to rubber stamps, civil society, independent media and the moral force of religious and academic institutions can emerge as the last formidable strongholds against tyranny. After all, sooner or later, dictators die, whereas collective efforts remain."
    # text ="failed vote oust president shakes peru's politics lima (reuters) - peru’s president pedro pablo kuczynski could end surprise winner attempt oust power week, opposition lawmakers broke ranks party leaders support him, opening divide might strengthen hand. despite congressional majority, rightwing opposition party popular force unable push motion remove kuczynski office thursday, 10 lawmakers broke ranks save president. vote cemented growing divide opposition looked threaten control congress, potentially aiding kuczynski tries restore political stability revive investments one latin america’s robust economies. surprise defection result deal struck kuczynski popular force rebel lawmaker kenji fujimori get father ex-president alberto fujimori prison, alleged popular force secretary general, jose chlimper. past year, kenji courted kuczynski’s center-right government challenging sister keiko’s leadership rightwing populist movement father formed 1990s. defiance sister, kenji threw support behind kuczynski ahead vote whether remove office unproven graft allegations. nine popular force lawmakers followed lead. “this birth serious formal split (in fujimori movement),” said guillermo loli, head political research pollster ipsos peru. “everything points pardon,” added.     kuczynski’s government denied pardon fujimori part political negotiations. address nation late friday, kuczynski said would spend coming days reflecting year half office. “i’ll announcing changes make sure 2018 year greater growth, politically different,” kuczynski said. efforts reach popular force lawmakers defected successful. one, clayton galvan, said local tv channel canal n alberto fujimori called prison ask help kuczynski stay power.     alberto fujimori, serving 25-year sentence graft human rights crimes, deeply divisive figure peru. many consider corrupt dictator, others credit ending economic crisis bloody leftist insurgency 1990-2000 term. freeing would likely anger well-organized foes fujimori clan - mix technocrats, leftists, human rights activists academics. “the day (kuczynski) signs pardon, loses guys. permanently,” said harvard university political scientist steve levitsky. support anti-fujimori crowd key kuczynski’s razor-thin victory keiko last year’s presidential election, keeping motion oust succeeding. “kuczynski saved two diametrically opposed political groups: kenji’s group left, opposes pardon. can’t please them,” said levitsky. kuczynski, 79-year-old former investment banker, took office amid hopes would usher cleaner government faster economic growth. instead, graft scandal roiling latin america stalled investments ensnared allegations wrongdoing. vote thursday, kuczynski fanned fears return peru’s authoritarian past described motion part legislative “coup” attempt keiko’s supporters. popular force denies charge says bid remove part fight corruption within bounds constitution. hardline popular force lawmaker loyal keiko, hector becerril, said kenji faction represented “traitors.”  “if sense decency vote, least could present resignations,” becerril told journalists friday. “hopefully today.” 10 votes fewer, popular force would command 61 seats 130-member, single-chamber congress, less absolute majority, though would still biggest voting bloc. political crisis cost kuczynski interior minister, carlos basombrio, announced resignation friday. kuczynski could make decision cabinet changes coming days, government said."
    #text = "The River Seine in Paris has reopened publicly to swimmers for the first time since 1923 after a century-long ban. The seasonal opening of the Seine for swimming is viewed as a key legacy of the Paris 2024 Olympics, when open water swimmers and triathletes competed in its waters which were specially cleaned for the event. On Saturday morning at 08:00 local time (07:00 BST) a few dozen swimmers arrived ahead of the opening and dived into the water when they were able to do so. There are three designated areas for public swimming in the Seine - one near the Eiffel Tower, another close to Notre Dame Cathedral and a third in eastern Paris. Zones have changing rooms, showers, and beach-style furniture, which allow for up to 300 people to lay out their towels. Until the end of August, the three swimming sites will be open for free at scheduled times to anyone with a minimum age of 10 or 14 years, depending on the location. Lifeguards will also be present keeping an eye on those in the river. The promise to lift the swimming ban dates back to 1988, when then-mayor of Paris and future president Jacques Chirac first advocated for its reversal. Improvements over the last 20 years have already led to a sharp reduction in faecal bacteria entering the river. Paris to bring back swimming in Seine after 100 years Would you swim in the Seine? For 100 years swimming was banned in the river because of the levels of water pollution that could make people ill. Ahead of last summer's Olympics more than €1.4bn (£1.2bn; $1.6bn) was invested into cleaning up the Seine. But, in the lead up to the games there were doubts as to whether the River Seine would be ready for the Olympics after it was revealed it failed water quality tests. Organisers blamed rainfall for the increased pollution which limited athletes' abilities to train for the triathlon, marathon swimming and paratriathlon. Last July, Paris Mayor Anne Hidalgo and other members of the Olympic committee went into the Seine to prove that it was safe to swim in."
#     text = (
#     "Europe is grappling with an early summer heat wave, leading to significantly low levels in major rivers. "
#     "The Danube in Hungary is at unusually low levels—only 17% of average June rainfall—with cargo ships forced to operate "
#     "at 30–40% capacity, triggering steep freight surcharges. The Vistula in Poland and the Rhine in Germany are also running dry. "
#     "Budapest hit 35 °C (95 °F), contributing to at least eight heat-related deaths this week. Hungary’s meteorological institute reported "
#     "June was the driest since 1901. Shipping disruptions and a spike in transport costs by as much as 100% are being observed, along with "
#     "negative effects on local ecosystems and agricultural communities along the rivers."
# )
    label = predict_news(text)
    print(label)