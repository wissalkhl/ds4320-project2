import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
import os

# load environment variables from .env file
load_dotenv()

# configure logging to write to logs folder
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/create_data.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# swing states we care about
SWING_STATES = {
    'PENNSYLVANIA': 'PA',
    'MICHIGAN': 'MI',
    'WISCONSIN': 'WI',
    'ARIZONA': 'AZ',
    'NEVADA': 'NV',
    'GEORGIA': 'GA',
    'NORTH CAROLINA': 'NC'
}

ELECTION_YEARS = [1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020]

def load_unemployment(data_dir):
    """Load all state unemployment CSVs and combine into one DataFrame."""
    try:
        frames = []
        for abbr in SWING_STATES.values():
            path = os.path.join(data_dir, f'{abbr}UR.csv')
            df = pd.read_csv(path, parse_dates=['observation_date'])
            df = df.rename(columns={f'{abbr}UR': 'unemployment_rate'})
            df['state_po'] = abbr
            frames.append(df)
        combined = pd.concat(frames)
        logging.info(f"Loaded unemployment data: {len(combined)} rows")
        return combined
    except Exception as e:
        logging.error(f"Failed loading unemployment: {e}")
        raise

def load_election_results(path):
    """Load presidential election CSV, filter to swing states and major parties."""
    try:
        df = pd.read_csv(path)
        df['state'] = df['state'].str.upper()
        df = df[df['state'].isin(SWING_STATES.keys())]
        df = df[df['party_simplified'].isin(['DEMOCRAT', 'REPUBLICAN'])]
        df = df[df['writein'] == False]
        logging.info(f"Loaded election data: {len(df)} rows after filtering")
        return df
    except Exception as e:
        logging.error(f"Failed loading election data: {e}")
        raise

def get_election_unemployment(ur_df, state_abbr, year):
    """Get average unemployment in Oct/Nov of election year for a given state."""
    try:
        mask = (
            (ur_df['state_po'] == state_abbr) &
            (ur_df['observation_date'].dt.year == year) &
            (ur_df['observation_date'].dt.month.isin([10, 11]))
        )
        avg = ur_df[mask]['unemployment_rate'].mean()
        return round(float(avg), 2) if not pd.isna(avg) else None
    except Exception as e:
        logging.error(f"Error getting unemployment for {state_abbr} {year}: {e}")
        return None

def build_documents(elec_df, ur_df):
    """Build one MongoDB document per state per election year."""
    documents = []
    try:
        for year in ELECTION_YEARS:
            year_df = elec_df[elec_df['year'] == year]
            for state_name, abbr in SWING_STATES.items():
                state_df = year_df[year_df['state'] == state_name]
                if state_df.empty:
                    logging.warning(f"No election data for {state_name} {year}, skipping")
                    continue

                total = int(state_df['totalvotes'].iloc[0])
                votes = {}
                for _, row in state_df.iterrows():
                    party = row['party_simplified'].lower()
                    votes[party] = int(row['candidatevotes'])

                if 'democrat' not in votes or 'republican' not in votes:
                    logging.warning(f"Missing party data for {state_name} {year}, skipping")
                    continue

                dem_share = round(votes['democrat'] / total * 100, 2)
                rep_share = round(votes['republican'] / total * 100, 2)
                winner = 'DEMOCRAT' if votes['democrat'] > votes['republican'] else 'REPUBLICAN'
                unemployment = get_election_unemployment(ur_df, abbr, year)

                doc = {
                    "year": year,
                    "state": state_name,
                    "state_po": abbr,
                    "democrat_votes": votes['democrat'],
                    "republican_votes": votes['republican'],
                    "total_votes": total,
                    "democrat_vote_share": dem_share,
                    "republican_vote_share": rep_share,
                    "two_party_margin": round(dem_share - rep_share, 2),
                    "winner": winner,
                    "unemployment_rate_election_season": unemployment,
                    "data_sources": ["MIT Election Lab", "FRED"]
                }
                documents.append(doc)
                logging.info(f"Built document: {state_name} {year} - winner: {winner}")

        logging.info(f"Total documents built: {len(documents)}")
        return documents
    except Exception as e:
        logging.error(f"Error building documents: {e}")
        raise

def insert_to_mongo(documents, uri):
    """Insert all documents into MongoDB Atlas collection."""
    try:
        client = MongoClient(uri)
        db = client["swing_state_elections"]
        col = db["election_economics"]
        col.delete_many({})  # clear old data before reinserting
        result = col.insert_many(documents)
        logging.info(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
        print(f"Successfully inserted {len(result.inserted_ids)} documents")
        client.close()
    except Exception as e:
        logging.error(f"MongoDB insert failed: {e}")
        raise

if __name__ == "__main__":
    MONGO_URI = os.getenv("MONGO_URI")
    DATA_DIR = "data/raw"

    print("Loading unemployment data...")
    ur_df = load_unemployment(DATA_DIR)

    print("Loading election results...")
    elec_df = load_election_results(os.path.join(DATA_DIR, "1976-2020-president.csv"))

    print("Building documents...")
    docs = build_documents(elec_df, ur_df)
    print(f"Built {len(docs)} documents")

    print("Inserting into MongoDB...")
    insert_to_mongo(docs, MONGO_URI)
    print("Done")