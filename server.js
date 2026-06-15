const express = require('express');
const mysql = require('mysql2');
const app = express();
const port = process.env.PORT || 3002;
app.use(express.static('public'));
const connection = mysql.createConnection({
  host: process.env.DB_HOST || 'localhost',
  database: process.env.DB_NAME || 'COMPANY_FINANCIALS',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
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

  
