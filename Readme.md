# MADS Capstone - Twitter Content Filter Chrome Extension API
The backend of the Twitter Content Filter Chrome Extension, built with fastapi and deployed on the Google Cloud 
Platform service "Cloud Run". Handles all preprocessing of Tweet content and classifying the sentiment of a Tweet to 
determine whether it should be filtered based on the inputs provided by the Chrome Extension user. Inputs are specific 
words to filter and a threshold that allows the user to decide for themselves how much negativity they are willing to 
tolerate.

## Authors
- [Dylan Kinder](https://github.com/drkinder)
- [Michael Philips](https://github.com/mphillipsjr96)
- [Ryan Maloney](https://github.com/rmaloney820)

## API Documentation

### Root URL
https://content-filter-api-js23pan5iq-uc.a.run.app

### Current Best Model - /filter-twitter-content/

#### Example cURL Request
```curl
curl --location --request POST 'https://content-filter-api-js23pan5iq-uc.a.run.app/filter-twitter-content/' \
--header 'Content-Type: application/json' \
--data-raw '{
    "body": "Here is an example of a Tweet's text",
    "threshold": 0.5,
    "filter_words": ["These", "are", "words", "to", "filter", "Tweets", "containing"]
}'
```
#### Example JSON Response
```
{
    "filter": boolean,  # should the tweet be filtered?
    "confidence_positive": float  # optional - if model.predict is called, the predicted positivity of the tweet 0-1
}
```
