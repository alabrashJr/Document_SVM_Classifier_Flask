# Document_SVM_Classifier_Flask

In different Terminal run,
```
cd svm/
docker-compose up 
```

```
python classifier.py <filenmae.txt>
```

## SVM_FLASK_Usage

All responses will have the form

```json
{ 
    'identifier':id of document (it could be the filename),
    'text':document's text

}
```

Subsequent response definitions will only detail the expected value of the `data field`

### List all queries

**Definition**

`GET /queries`

**Response**

- `200 OK` on success

```json
[
    {
        "identifier": "5.html",
        "text": "......."
        "output": 0
    },
    {
        "identifier": "demo1.html",
        "text": "......",
        "output": 1
    }
]
```

### Registering a new query

**Definition**

`POST /queries`

**Arguments**

- `"identifier":string` a globally unique identifier for this query
- `"text":string` the document's text


If a query with the given identifier already exists, the existing query will be overwritten.

**Response**

- `201 Created` on success

```json
{
    "identifier": "5.html",
    "text": "......."
    "output": 0
}
```

## Lookup query details

`GET /queries/<identifier>`

**Response**

- `404 Not Found` if the query does not exist
- `200 OK` on success

```json
{
    "identifier": "5.html",
    "text": "......."
    "output": 0
```

## Delete a query

**Definition**

`DELETE /queries/<identifier>`

**Response**

- `404 Not Found` if the query does not exist
- `204 No Content` on success
