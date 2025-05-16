import type { NextApiRequest, NextApiResponse } from 'next'
const fs = require('fs');
 
type ResponseData = {
  message: string | null;
}


export default function handler(req: NextApiRequest, res: NextApiResponse<ResponseData>) {
  const { user } = req.query;
  const { article } = req.query;
  const fileName = `data/annotations/` + user + "_" + article;
  try {
    const content = fs.readFileSync(fileName, "utf8");
    res.status(200).json({ message: JSON.parse(content) })
  } catch (error) {
    res.status(404).json({message: null});
  }

}