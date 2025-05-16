import type { NextApiRequest, NextApiResponse } from 'next'
const fs = require('fs');
 
type ResponseData = {
  message: string
}



export default function handler(
    req: NextApiRequest,
    res: NextApiResponse<ResponseData>
  ) {
    // TODO: need to send a complex object back here. It's all bytes.
    fs.readdir('data/articles', (err, files) => {
        if (err)
          console.log(err);
        else {
          console.log("\nCurrent directory filenames:");
          
          res.status(200).json({ message: JSON.stringify(files) })
        }
    });
  }