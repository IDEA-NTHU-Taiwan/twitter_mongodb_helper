##  Twitter MongoDB helper functions
A collection of python functions for dealing with mongoDB indexes, aggregation pipleline and map-reduce

## MongoDB Indexing

It is essential to create indexes on your fields in order to speed of the run time of your queries. A few ideal indexes will be outlined below but ultimately it will depend on your data needs.

### Resources

- [Text Search](https://docs.mongodb.com/manual/text-search/)
- [Create Text Index](https://docs.mongodb.com/manual/core/index-text/#create-a-text-index)
- [Multi-Key Indexes](https://docs.mongodb.com/manual/core/index-multikey/)
- [Compound Prefix](https://docs.mongodb.com/manual/core/index-compound/#compound-index-prefix)
- [Multi-Key Indexes](https://docs.mongodb.com/manual/core/index-multikey/)
- [Manging Indexes](https://docs.mongodb.com/v3.2/tutorial/manage-indexes/)

### Instructions

This assumes that you already have MongoDB installed and running and already populated with tweets.

Start the mongo shell from the terminal and select your databse

```bash
mongo
use twitterdb
```

### Creating a text index

```bash
db.tweets.createIndex({text:"text","entities.hashtags.text":"text"}, {background:true})
```

This allows for 'Google search' like capability on the hashtags and the text of the tweet itself. Only one text index can exist at a time.

```bash
db.tweets.find({$text:{$search: "java shop coffee"}})
```
### Creating a compound index

```bash
db.tweets.createIndex({"user.id": 1, "user.statuses_count":1, "user.followers_count":1, "user.location":1, "user.lang":1}, {background:true})
```

### Creating a single field index

```bash
db.tweets.createIndex({timestamp_ms: 1}, {background:true})
db.tweets.createIndex({"entities.user_mentions": 1}, {background:true})
db.tweets.createIndex({"entities.urls": 1}, {background:true})
```

### Fun fact

Aggregation is orders of magnitude faster than map reduce
