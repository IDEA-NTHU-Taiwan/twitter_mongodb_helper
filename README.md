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
db.tweets.createIndex({"lang": 1}, {background:true})
db.tweets.createIndex({"entities.urls": 1}, {background:true})
```

### Insights

- Aggregation is orders of magnitude faster than map reduce
- Queries that scan the full database may time out when executed at the application level (PyMongo etc). If this happens try running them from Mongo Shell instead

### Aggregation vs MapReduce

Get an aggregate count of all the hashtags in a collection, filtered by english. (uses mongo shell)

```bash
db.tweets.aggregate([{$match: {'lang': {$in: ['en']}}}, {$project: {'entities.hashtags': 1, _id : 0}}, {$unwind: '$entities.hashtags'}, {$group: {_id: '$entities.hashtags.text', count: {$sum: 1}}}, {$sort: {count: -1}}, {$project: {"hashtag": "$_id", "count": 1, "_id": 0}}, { $out : "hashtag_dist_en" }])
```

Same query with MapReduce (Pymongo)

```python
def hashtag_map_reduce(client, db_name, subset, output_name):
    map_function = Code("function () {"
                        "    var hashtags = this.entities.hashtags;"
                        "    for (var i = 0; i < hashtags.length; i ++){"
                        "        if (hashtags[i].text.length > 0) {"
                        "            emit (hashtags[i].text, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyHashtag, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name, query={"lang": {"$eq": 'en'}})
```

Aggregate user mentions, same structure as the above
```bash
db.tweets.aggregate([{$match: {'lang': {$in: ['en']}}}, {$project: {'entities.user_mentions': 1, _id: 0}}, {$unwind: '$entities.user_mentions'}, {$group: {_id: {id_str: '$entities.user_mentions.id_str', 'screen_name': '$entities.user_mentions.screen_name'}, count: {$sum: 1}}}, {$project: {id_str: '$_id.id_str', 'screen_name': '$_id.screen_name', 'count': 1, '_id': 0}}, {$sort: {count: -1}}, { $out : "user_mentions_dist_en" }])
```
