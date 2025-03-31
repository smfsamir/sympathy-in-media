import type { NextApiRequest, NextApiResponse } from 'next'
const fs = require('fs');
 
type ResponseData = {
  message: string | null;
}


export default function handler(req: NextApiRequest, res: NextApiResponse<ResponseData>) {
  const { user } = req.query;
  const { article } = req.query;
  if (!article) {
    return res.status(400).json({ message: null});
  }
  const fileName = `data/annotations/` + user + "_" + article;
  try {
    const content = fs.readFileSync(fileName, "utf8");
    res.status(200).json({ message: JSON.parse(content) })
  } catch (error) {
    res.status(500).json({message: null});
  }

}