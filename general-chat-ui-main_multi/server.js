const express = require('express')
const app = express()
app.use(express.static('.')) //serve our files in public statically
app.listen(3456)
