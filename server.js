const express = require('express');
const mysql = require('mysql2');
const app = express();
const port = 3002;
app.use(express.static('public'));
const connection = mysql.createConnection({
  host: 'localhost',
  database: 'COMPANY_FINANCIALS',
  user: 'root',
  password: 'dps#1234',
});
connection.connect((err) => {
  if (err) {
    console.error('Error connecting: ' + err.stack);
    return;
  }
  console.log('Connected as id ' + connection.threadId);
});
app.get('/COMPANY_FINANCIALS.company', (req, res) => {
  connection.query('SELECT * FROM COMPANY_FINANCIALS.company', (error, results) => {
    if (error) {
      res.status(500).send('Database query error');
    } else {
      res.json(results); 
    }
  });
});
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

  