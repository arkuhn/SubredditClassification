# Get Started

1. Install requirements

   `pip install -r requirements.txt`

2. Configure API credentials

   - Create an 'APIconfig.py' following the provided template with your credentials
     ```
     CLIENT_ID='your client id'

     CLIENT_SECRET='your client secret'

     USER_AGENT='your user agent string'

     USERNAME='your username'

     PASSWORD='your password'
     ```

3. Configure targeted subreddits

   - Modify subredditsConfig.py to desired targets

     ```
     SUBREDDIT1='subreddit name 1'

     SUBREDDIT2='subreddit name 2'

     POSTCOUNT= how many posts of each to pull
     ```

4. Crawl data

   `python dataCrawl.py`

5. Run a models

   `python main.py`

