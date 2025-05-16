import type { NextApiRequest, NextApiResponse } from 'next'
const fs = require('fs');
 
type ResponseData = {
  message: string
}
 
export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseData>
) {
  fs.writeFileSync('data/users.json', JSON.stringify(['BRUH'], null, 4)); // TODO: need to update this.
  // TODO: need to send a complex object back here. It's all bytes.
  res.status(200).json({ message: 'Hello from Next.js!' })
}